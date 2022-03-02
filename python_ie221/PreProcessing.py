import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

class PreProcessing:
    """Process DataFrame with many steps:
        1. Fill missing data.
        2. Remove any special characters : &$^@!)$*% ... 
        3. lower character :  ABC -> abc
        4. remove stop words : a, an, the, this (cause it has no value for trainning)
        5. combine 25 title to a column name "filter"
        6. split label as y, filter as x  
        7. split x and y to x_train, x_test, y_train, y_test with size 0.2 (20% for test, 80% for train)
    
         
    Attributes:
        pre_data(DataFrame): with 3 columns  'Date', 'Label', 'Filter'
        x_train(list)
        x_test(list)
        y_train(list)
        y_test(list)
    """
    def __init__(self):
        pass
            
    def count_null(self, data):
        """Count how many null in your data
        
        Args:
            data(dataFrame): your data after convert from csv, json, ...
        
        Returns:
            Series: Object with missing values
        
        """
        print("\nNumber of NaN in your data: ",data.isnull().sum().sum())
        return data.isnull().sum()
    
    def fill_null(self, data):
        """Find any null  data and replace it with ' ' in column or drop if it is 'LABEL'
        
        Args:
            data(dataFrame): your data with missing value
            
        Returns:
            dataFrame: data with no missing value
        """
        for col in data.columns:
            if(pd.api.types.is_string_dtype(data[col]) == True):
                data[col].fillna(' ', inplace = True)
        data = data.dropna()
        print("\nAfter fill, your number of nan data is ",data.isnull().sum().sum())
        
        self.pre_data = data
        return self.pre_data
    
    def remove_punc_and_lower(self, data):
        """Remove all punctuation and change letter to lower case (except Date and Label columns)
        
        Args:
            data(dataFrame): 
        
        Returns:
            dataFrame: data after change
        """
        
        for col in data.columns:
            if col == 'Date':
                continue
            if col == "Label":
                continue
            temp = data.loc[:,col].copy()
            data.loc[:,col] = temp.str.lower()
            data.loc[:,col].replace(["b"+"[^a-zA-Z]","[^a-zA-Z]"]," ",True,None,True)
        
        self.pre_data = data
        return self.pre_data
    
    def combine_title(self,data):
        """Combine all title to a column  
        
        Args: 
            data(dataFrame): 
        
        Returns:
            dataFrame: combined title data
        """
        head_line = []
        for i in range(0,len(data.index)):
            head_line.append(''.join(str(x) for x in data.iloc[i,2:-1]))
        
        data = data.drop(data.iloc[:,2:], axis=1)
        data['title'] = head_line
        
        self.pre_data = data
        return self.pre_data
    
    def remove_stopword(self,data):
        """Remove common word cause it has no value for our processing
        
        Args:
            data(dataFrame)
        
        Returns:
            dataFrame
        
        For ex:  "A and B is friend" -> ['A','B','friend']
        """
        
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)
        list_filter = []
        for row in range(0,data.shape[0]):
             fil = ' '.join([word for word in data.iloc[row,2].split() if word not in stopwords_dict])
             list_filter.append(fil)
        data['filter'] = list_filter
        
        data = data.drop(columns = 'title')
        
        self.pre_data = data
        return self.pre_data
    
    def split_x_y(self):
        
        self.x = self.pre_data.loc[:,'filter']
        self.y = self.pre_data.loc[:,'Label']
      
    def convert_train_test(self):
        """Convert data to train/ test set. Then convert it to LIST type for processing
        
        Args:
            self.pre_data(dataFrame): need a preprocessed data be like a PreProcessing attribute
       
        Returns:
            list: attribute x_train/test , y_train/test 
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.2)
        self.x_train.to_csv(r'data\train_test_split\x_train.csv')
        self.y_train.to_csv(r'data\train_test_split\y_train.csv')
        self.x_test.to_csv(r'data\train_test_split\x_test.csv')
        self.y_test.to_csv(r'data\train_test_split\y_test.csv')
        self.x_train = self.x_train.tolist()
        self.x_test = self.x_test.tolist()
        self.y_test = self.y_test.tolist()
        self.y_train = self.y_train.tolist()
        
        
    def fully_preprocess(self,data):
        """ PreProcess data with all step : fill null, remove stop word, combine. Finally, split data
        to x and y
        
        Args:
            data(dataFrame): original data
        
        Returns:
            dataFrame: preprocessed data and file csv in 'data\preprocessed_data.csv'
            create x,y train/test attribute
            
        """
        self.count_null(data)
        self.pre_data = self.fill_null(data)
        self.pre_data = self.remove_punc_and_lower(self.pre_data)
        self.pre_data = self.combine_title(self.pre_data)
        self.pre_data = self.remove_stopword(self.pre_data)
        final_csv = self.pre_data
        final_csv = final_csv.set_index('Date')
        final_csv.to_csv('data\preprocessed_data.csv')
        self.split_x_y()
        self.convert_train_test()
        return self.pre_data


