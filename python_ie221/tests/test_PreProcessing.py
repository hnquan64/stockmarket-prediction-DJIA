import pytest
import pandas as pd
from python_ie221.PreProcessing import PreProcessing
d = {'Date': ['1','2', '3', '4' ], 
     'Label': [0, 1, 0, 1], 
     'Top1': ['cat dog','monkey',float("NaN"),'horse cow'],
     'Top2': ['penguin','pig','fox', float("NaN")]
     }
d2 =  {'Date': ['1','2', '3', '4' ], 
     'Label': [0, 1, float("NaN"), 1], 
     'Top1': ['cat dog','monkey',float("NaN"),'horse cow'],
     'Top2': ['penguin','pig','fox', float("NaN")]
     }

d3 = {'top1': 'b"Cat!@#$', 'top2':'"This"dog'}

data = pd.DataFrame(d)
data2 = pd.DataFrame(d2)
data3 = pd.DataFrame(d3, index=[0])
class TestFillNull(object):
        def test_fill_null_miss_string(self):
            predict = pd.DataFrame({'Date': ['1','2', '3', '4' ], 
                                        'Label': [0, 1, 0, 1], 
                                        'Top1': ['cat dog','monkey',' ','horse cow'],
                                        'Top2': ['penguin','pig','fox', ' ']})
            
            actual = PreProcessing().fill_null(data)
            
            assert predict.equals(actual)
            
        def test_fill_null_miss_label(self):
            predict = pd.DataFrame({'Date': ['1','2', '4' ], 
                                        'Label': [0, 1, 1], 
                                        'Top1': ['cat dog','monkey','horse cow'],
                                        'Top2': ['penguin','pig', ' ']}).values
            
            actual = PreProcessing().fill_null(data2).values
            
            assert (predict == actual).all()
            

def test_remove_punc_and_lower():
    predict = pd.DataFrame({'top1':' cat    ','top2':' this dog'},index=[0]).values
    actual = PreProcessing().remove_punc_and_lower(data3).values
    assert (predict == actual).all()

