import pandas as pd

class DATA():
    def __init__(self, train_data=None, val_data=None):
        self.train_data = [train_data]
        self.val_data = [val_data]
        if self.val_data and self.train_data:
            self.df = pd.concat([train_data, val_data])
        elif self.train_data is not None:
            self.df = self.train_data
        elif self.val_data is not None:
            self.df = self.val_data
        else:
            self.df = pd.DataFrame()


    def add_train(self, data):
        if self.train_data:
            self.train_data = pd.concat([self.train_data, data])
        else:
            self.train_data = data
        if self.val_data:
            self.df = pd.concat([self.train_data, self.val_data])
        else:
            self.df = self.train_data

    def add_val(self, data):
        if self.val_data:
            self.val_data.append(data)
        else:
            self.val_data = data
        if self.train_data:
            self.df = pd.concat([self.train_data, self.val_data])
        else:
            self.df = self.val_data