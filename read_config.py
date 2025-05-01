"""Defines the configuration to be loaded before running any experiment"""
from configobj import ConfigObj
import string


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config

        # Model name and location to store
        self.model_path = config["train"]["model_path"]

        # path to the model
        self.pretrain_model_path = config["train"]["pretrain_model_path"]
        self.pre_model_path = config["train"]["pre_model_path"]
        self.pre_opt_model_path = config["train"]["pre_opt_model_path"]

        self.prefix = config["train"]["prefix"]
        # number of training examples
        self.num_train = config["train"].as_int("num_train")
        self.num_test = config["train"].as_int("num_test")
        self.num_points = config["train"].as_int("num_points")

        # Number of epochs to run during training
        self.epochs = config["train"].as_int("num_epochs")

        # batch size, based on the GPU memory
        self.batch_size = config["train"].as_int("batch_size")

        self.gpu = config["train"]["gpu"]
        # Mode of training, 1: supervised, 2: RL
        self.mode = config["train"].as_int("mode")
        self.edge_loss_method = config["train"].as_int("edge_loss_method")

        # Learning rate
        self.lr = config["train"].as_float("lr")
        self.eval_T = config["train"].as_int("eval_T")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())
