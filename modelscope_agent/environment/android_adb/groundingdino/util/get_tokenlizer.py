from transformers import (AutoTokenizer, BertModel, BertTokenizer,
                          RobertaModel, RobertaTokenizerFast)


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, 'text_encoder_type'):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get('text_encoder_type', False):
            text_encoder_type = text_encoder_type.get('text_encoder_type')
        else:
            raise ValueError('Unknown type of text_encoder_type: {}'.format(
                type(text_encoder_type)))
    print('final text_encoder_type: {}'.format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    return BertModel.from_pretrained(text_encoder_type)
