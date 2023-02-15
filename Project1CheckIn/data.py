'''data.py
Reads CSV files, stores data, access/filter data by variable name
Ryan Mogauro
CS 251 Data Analysis and Visualization
Spring 2023
'''


from email import header
import csv
import numpy as np


class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, datatypes=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        '''

        self.filepath = filepath
        self.datatypes = []
        self.headers = []
        self.data = []
        self.header2col = {}

        if filepath is not None:
            self.read(filepath)

        pass

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        '''

        if(True):
            skip = []
            with open(filepath, 'r') as csv_file:
                reader = csv.reader(csv_file)
                counter = 0

                for row in reader:

                    #if first row
                    if(counter == 0):
                        headerHolder = []
                        for i in row:
                          headerHolder.append(i.strip())
                        counter+=1
                        
                    
                    #if second row
                    elif(counter == 1):
                        newHeaders = []
                        newcount = 0
                        for i in row: 
                            self.datatypes.append(i)
                            if i.strip() != "numeric":
                                skip.append(newcount)
                
                            if i.strip() == "numeric": 
                                newHeaders.append(headerHolder[newcount])
                            newcount+=1
                        counter+=1

                        self.headers = newHeaders


                        headcounter = 0
                        for i in self.headers: 
                            self.header2col[i] = headcounter
                            headcounter+=1

                #if not first or second row
                    else:
                        rowData = []
                        rowCount = 0
                        for i in row: 
                            if (rowCount not in skip):
                                rowData.append(float(i))
                            rowCount+=1
                        self.data.append(rowData)

        #turn self.data into Numpy array
        self.data = np.array(self.data)

        #returns None
        return None

    def __str__(self):
        #toString method
        '''
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        result = "Headers: \n"
        for i in self.headers:
            result+=str(i)+"\t"
        result += "\nShowing rows 5 / " + str(len(self.data)) + "\n"
        
        for i in range(0,5):
            if(i < len(self.data)):
                for data in self.data[i]:
                    result+=str(data)
                    result+="\t"
                result+="\n"
        
        return result
        

    def get_headers(self):
        #header get method
        #returns list of headers
        return self.headers

    def get_mappings(self):
        #returns mapping between variable name and column index
        return self.header2col

    def get_num_dims(self):
        #returns number of variables
        numVars = 0
        for i in self.data[0]:
            numVars+=1
        return numVars

    def get_num_samples(self):
        #returns number of datum in the data
        return len(self.data)

    def get_sample(self, rowInd):
        #returns row at rowInd parameter
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:=
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        indices = []
        
        for i in headers: 
            indices.append(self.headers.index(i))
        return indices


    def get_all_data(self):
        #returns copy of data field, representing all data
        datacopy = self.data.copy()
        return datacopy
        pass

    def head(self):
        '''Return the 1st five data samples (all variables)

        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        head = ""
        for i in self.data:
            for datum in i:
                head+= str(datum) + "\t"
        return head
        
        pass

    def tail(self):
        '''Return the last five data samples (all variables)
        ndarray. shape=(5, num_vars). Last five data samples.
        '''

        tail = ""
        counter = 0
        for i in self.data:
            if((len(self.data) - counter) > 5):
                for datum in i:
                    tail+= str(i) + "\t"
            counter+=1
        return tail    
        pass

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''

        newdata = []
        counter = 0
        for i in self.data:
            if(counter >= start_row and counter < end_row):
                rowdata = []
                for datum in i:
                    rowdata.append(i)
                newdata.append(rowdata)
        self.data = np.array(newdata)
        pass

    def select_data(self, headers, rows=[]):

        #use numpy.ix supposedly 

        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''

        
        pass
