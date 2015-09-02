from data_process import *
import RNN_LM as model
import sys
if __name__ == "__main__":

    if (len(sys.argv)==4):
        training = sys.argv[1]
        validation = sys.argv[2]
        test = sys.argv[3]
    else:
        print "Arguments: must have three!"
        sys.exit(1)
    dataset = generate_dataset(training, validation, test)
    myrnn = model.RecurrentNet(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4],epoch=20,learnrate=0.1,hidden_size=15)
    myrnn.nntrain()
    myrnn.nntest()
