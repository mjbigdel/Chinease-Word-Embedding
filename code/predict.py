from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib import crf  
from PredictFunctions import ngrams_pretrain_embed,create_tensorflow_model, get_test_data, padding3, ids_to_tag_file

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """    
    ngrams2id, word_embeddings, PAD_ID = ngrams_pretrain_embed(resources_path)
    VOCAB_SIZE =  word_embeddings.shape[0]
    WORD_EMBEDDING_DIM = word_embeddings.shape[1]
    
    BATCH_SIZE = 16
    HIDDEN_LAYER_DIM = 128
    
    test_x = get_test_data(input_path,ngrams2id, usebigram=True, usetrigram=True,usefourgram=True)
    
    X, labels, output, train_op, dropout_keep_prob, loss, transition_params, seq_length = create_tensorflow_model(VOCAB_SIZE, WORD_EMBEDDING_DIM, HIDDEN_LAYER_DIM, word_embeddings, PAD_ID)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        saver.restore(sess, resources_path+'/model.ckpt')        

        test_pred = []
        for i in range(0, len(test_x), BATCH_SIZE):
            batch_x = test_x[slice(i, i + BATCH_SIZE)]
            batch_x = padding3(batch_x, PAD_ID)
    
            lengths,unary_scores,transition_param_other = sess.run(
                [seq_length,output,transition_params], feed_dict = {X:batch_x, dropout_keep_prob:1.0})
            predict=[]
            for unary_score,length in zip(unary_scores,lengths):
                if length > 0 :
                    viterbi_sequence, _=crf.viterbi_decode(unary_score[:length],transition_param_other)
                    predict.append(viterbi_sequence)
                else:
                    predict.append("")                
            test_pred += predict          
    
    ids_to_tag_file(test_pred, output_path)
    
    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
