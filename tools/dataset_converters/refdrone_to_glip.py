from pycocotools.coco import COCO
import nltk
from nltk import pos_tag, word_tokenize
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def analyze_phrase_from_tokens(caption, tokens_positive):
    """
    分析tokens_positive对应的短语，提取主语和对应的token位置
    
    Parameters:
    caption: 完整的描述文本
    tokens_positive: token的位置列表 [[start1, end1], [start2, end2], ...]
    
    Returns:
    subject_info: 包含主语及其对应token位置的字典
    """
    # 获取完整的短语
    words = caption.split()
    phrase_words = []
    token_positions = []
    
    # 收集所有token对应的词和位置
    for start, end in tokens_positive:
        phrase_words.extend(words[start:end])
        token_positions.extend([(start, end)])
    
    phrase = ' '.join(phrase_words)
    
    # 使用NLTK进行词性标注
    tokens = word_tokenize(phrase)
    pos_tags = pos_tag(tokens)
    
    # 查找主语（通常是第一个名词或名词短语）
    subject_words = []
    subject_positions = []
    
    current_position = 0
    for word, tag in pos_tags:
        # 检查是否是名词相关的标签
        if tag.startswith(('NN', 'DT')):  # NN:名词, DT:限定词
            subject_words.append(word)
            # 找到对应的token position
            for start, end in token_positions:
                if current_position >= start and current_position < end:
                    if (start, end) not in subject_positions:
                        subject_positions.append((start, end))
        current_position += 1
        
        # 如果遇到动词或其他非名词性词，就停止
        if tag.startswith('VB'):
            break
    
    return {
        'subject': ' '.join(subject_words),
        'token_positions': subject_positions
    }

if __name__ == "__main__":
    ann_file = '/mnt/public/usr/sunzhichao/RefDrone/finetune_RefDrone_test6.json'
    coco =COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        raw_img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annos = coco.loadAnns(ann_ids)[0]
        print(annos)
        caption = raw_img_info['caption']
        token_positive = annos['tokens_positive']
        result = analyze_phrase_from_tokens(caption, token_positive)
        print(f"Original caption: {caption}")
        print(f"Subject phrase: {result['subject']}")
        print(f"Token positions: {result['token_positions']}")

        print(raw_img_info)
        print(ann_ids, annos)
        exit()