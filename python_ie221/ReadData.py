#!/usr/bin/python
#-*- coding: utf-8 -*-

import pandas as pd
from pandas.io.parsers import read_csv

class ReadData:
    """This read data from csv and convert to dataFrame
    
    Ex:
        Format of csv file must have:
        Date(datetime)   Label(int)  Top1(string) Top2(string) .... Top25(string) *top1->top25
        
    Attributes:
        data(DataFrame)
    """
    
    def __init__(self):
        pass
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)

    
    def read_data(self, filepath):
        """Read data from file csv

        Args: 
            Filepath(string) : file data kiểu csv

        Returns: 
            dataFrame
        """
        self.data = pd.read_csv(filepath)
        return self.data
    
