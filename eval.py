import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from model import AtJ, Encoder

from rouge import *
from cider import *

# Parameters
data_folder = 'E:\\Image_Captioning_data\\haze'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'C:\\pung\\image_captioning\\BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
encoder_checkpoint = 'C:\\pung\\derain_imagecaption\\dehaze_patch_encoder\\epochs_patch_Ats_encoder_target_8000\\encoder_300.pth'
word_map_file = 'E:\\Image_Captioning_data\\haze_remove2\\J_image\\WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

#pretrained_weight = 'C:\\pung\\derain_imagecaption\\dehaze_patch_encoder\\epochs_patch_Ats_encoder_target_8000\\netG_epoch_300.pth'
# pretrained_weight = 'C:\\pung\\derain_imagecaption\\dehaze_patch\\epochs_patch_Ats\\netG_epoch_498.pth'
pretrained_weight = 'C:\\pung\\derain_imagecaption\\dehaze_patch\\epochs_patch_Ats\\netG_epoch_201.pth'

# Load model
checkpoint = torch.load(checkpoint)
#encoder_checkpoint = torch.load(encoder_checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
#encoder.info()
encoder = encoder.to(device)
encoder.eval()

netG = AtJ()
pretrained_weight = torch.load(pretrained_weight)
netG.load_state_dict(pretrained_weight)
netG = netG.to(device)

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    gts = {}
    res = {}
    name_number = 0
    
    references_bleu = list()
    hypotheses_bleu = list()

    references_meteor = list()
    hypotheses_meteor = list()

    count = 0
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)


        J, A, t, s = netG(image)

        # img = tensor_to_image(J)
        # img = np.clip(img * 255, 0, 255)
        # img = Image.fromarray(img.astype(np.uint8))
        # img.show()
        #img = img.astype(np.uint8)
        #img = img.transpose((2, 0, 1))
        #img = torch.FloatTensor(img / 255.)
        ##transform=transforms.Compose([normalize])
        #img = transform(img)
        #img = img.unsqueeze(0).cuda()
        
        # Encode
        encoder_out = encoder(J)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads

        # bleu
        references_bleu.append(img_captions)
        hypotheses_bleu.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        assert len(references_bleu) == len(hypotheses_bleu)

        # meteor
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  
                
        references_meteor.append(ch_string_references(img_captions))
        hypotheses_meteor.append(" ".join([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]))
        assert len(references_meteor) == len(hypotheses_meteor)

        # rouge_cider
        gts[str(name_number)] = img_captions
        res[str(name_number)] = [" ".join([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])]
        name_number += 1
        
        # count += 1
        # if count == 5:
        #     break

    # Calculate BLEU scores
    print("bleu-1 : ", corpus_bleu(references_bleu, hypotheses_bleu, weights=(1, 0, 0, 0)))

    print("bleu-2 : ", corpus_bleu(references_bleu, hypotheses_bleu, weights=(0.5, 0.5, 0, 0)))

    print("bleu-3 : ", corpus_bleu(references_bleu, hypotheses_bleu, weights=(0.33, 0.33, 0.33, 0)))

    print("bleu-4 : ", corpus_bleu(references_bleu, hypotheses_bleu, weights=(0.25, 0.25, 0.25, 0.25)))

    # Calculate METEOR scores
    total = 0
    for re, hy in zip(references_meteor, hypotheses_meteor):
        met = meteor_score(re, hy)
        total += round(met,4)
        #print(met)
    print("meteor : ", total/len(references_meteor))

    # Calculate ROUGE, CIDER scores
    rouge = Rouge()
    cider = Cider()
    print("rouge score : ", rouge.compute_score(gts, res))
    print("cider score : ", cider.compute_score(gts, res))


def ch_string_references(list):
    for line in range(len(list)):
        list[line] = " ".join(list[line])
    return list

def tensor_to_image(tensor):
    if type(tensor) in [torch.autograd.Variable]:
        img = tensor.data[0].cpu().detach().numpy()
    else:
        img = tensor[0].cpu().detach().numpy()
    img = img.transpose((1,2,0))
    #try:
    img = np.clip(img, 0, 255)
    if img.shape[-1] == 1:
        img = np.dstack((img, img, img))
    # except:
    #     #print("invalid value catch")
    #     Image.fromarray(img).save('catch.jpg')
    return img

if __name__ == '__main__':
    beam_size = 5
    evaluate(beam_size)
    #print("\nBLEU-1,2,3,4 score @ beam size of %d is %.4f, %.4f, %.4f." % (beam_size, evaluate(beam_size)))
