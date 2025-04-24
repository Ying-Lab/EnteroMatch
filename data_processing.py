import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import alpha_diversity

class data_process:
    def __init__(self,path,title,std=True,taxonomy=True,threshold=0.95,frac=0.5):
        self.path=path
        self.otu_table=pd.read_csv(path,index_col=0)
        self.title=title  # restore the names of self. or recipient
        self.std=std # default is True
        self.taxo=taxonomy # drop taxonomy
        self.threshold=threshold
        self.frac=frac


    def data_proces(self):
         # change row names and found a list
         row_names = [f'{self.title} otu {i}' for i in range(1, len(self.otu_table) + 1)]  
         self.otu_table.index = pd.Index(row_names)

         # drop NAN
         self.otu_table=self.otu_table.dropna(axis=1)
         self.otu_table=self.otu_table.drop_duplicates()

         if self.taxo==True:
             self.otu_table=self.otu_table.drop(index=self.otu_table.index[-1])
             

         # drop column containing value less than zero
         self.otu_table = self.otu_table.loc[:, (self.otu_table >= 0).all()]
         self.otu_table=self.otu_table.T
        
         self.otu_table=self.otu_table.drop_duplicates()
         #delete similar data
         similarity_matrix = cosine_similarity(self.otu_table)
         np.fill_diagonal(similarity_matrix,0)
         similarity_matrix=pd.DataFrame(similarity_matrix,index=self.otu_table.index,columns=self.otu_table.index)
         high_similarity_pairs = [
            (similarity_matrix.index[i], similarity_matrix.columns[j])
            for i, j in np.argwhere(similarity_matrix.to_numpy() > self.threshold)
                                ]
         samples_to_remove = set()
         for i, j in high_similarity_pairs:
        # 只保留一个样本，去掉重复的
            if i not in samples_to_remove and j not in samples_to_remove:
                    samples_to_remove.add(j)
         self.otu_table = self.otu_table.drop(index=list(samples_to_remove))
         if self.title=='recipient':
            self.otu_table=self.otu_table.sample(frac=self.frac,random_state=42)

         # add alpha diversity
         self.alpha_diversity=alpha_diversity.AlphaDiversityCalculator(self.otu_table)
         self.alpha_diversity.calculate_alpha_diversity_metrics()
         self.diversity_table=self.alpha_diversity.get_results()
         # drop  features' variance equal to zero
         self.diversity_table = self.diversity_table.loc[:, self.diversity_table.var() != 0]
         self.diversity_table=self.diversity_table.add_prefix(f'{self.title} ')
         
         # standardlize alpha diversity and
         if self.std==True:
             scaler_ = StandardScaler()
             standard_diversity_pd1=scaler_.fit_transform(self.diversity_table)
             self.diversity_table=pd.DataFrame(standard_diversity_pd1,columns=self.diversity_table.columns,index=self.diversity_table.index)
        
         self.otu_sum=self.otu_table.sum(axis=1)

         self.otu_table=self.otu_table.div(self.otu_sum,axis=0)

         self.feature=pd.concat([self.otu_table,self.diversity_table],axis=1)

         
    def alpha_features(self):
        return self.diversity_table
    def get_alpha_table(self,alpha_table):
        self.alpha_table=alpha_table

    def proce_otu_table(self):
        return self.otu_table
    
    def get_otu_table(self,otu_table):
        self.otu_table=otu_table
    
    # Return donor or recipient features
    def features_table(self):
        return self.feature
    

    # Combine donor features and recipient features. Default: self.feature:donor , recipient_feature: recipient
    def combined_table(self,recipient_feature):

        # duplicate donor features
        n=len(recipient_feature.index)
        duplicate_donor_feature = pd.concat([self.feature.loc[[ind]] for ind in self.feature.index for _ in range(n)], axis=0)

        # update column names
        duplicate_donor_feature.index = np.repeat(self.feature.index, n)

        # duplicate recipient features
        duplicate_recipient_feature=pd.concat([recipient_feature]*len(self.feature.index),axis=0)

        # store recipient features columns
        keep_recipient_index=duplicate_recipient_feature.index

        duplicate_recipient_feature.index=duplicate_donor_feature.index

        donor_recipient_feature=pd.concat([duplicate_donor_feature,duplicate_recipient_feature],axis=1)
        donor_recipient_feature.index = [f"{donor_recipient_feature.index[i]}-{keep_recipient_index[i]}" for i in range(len(donor_recipient_feature.index))]
        
        donor_recipient_feature.loc[:,'donor']=duplicate_donor_feature.index
        donor_recipient_feature.loc[:,'recipient']=keep_recipient_index
        return donor_recipient_feature

        
# Add label
class label_table:
    def __init__(self,path,thres=0.01):
        self.path=path
        self.label=pd.read_csv(self.path,index_col=0)
        self.thres=thres

    def label_process(self):
        self.label=self.label[self.label>=0]
        self.label=self.label.dropna(axis=1,how='all')
        self.label.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.label=self.label.dropna(axis=0,how='all')


        # Set nan into -1
        self.label.fillna(-1,inplace=True)

        # Set threshold 
        self.label[self.label>self.thres]=0
        self.label[(self.label<=self.thres)&(self.label>0)]=1

        
    # match donor_recipient and label
    def add_label(self,donor_recipient):

        label_flat = self.label.T.stack().reset_index()
        label_flat.columns = ['donor', 'recipient', 'label']
        label_flat['recipient']=label_flat['recipient'].astype(str)
        donor_recipient['recipient'] = donor_recipient['recipient'].astype(str)
        self.data = donor_recipient.merge(label_flat, on=['donor', 'recipient'], how='left')
        self.data=self.data.dropna(axis=0)
        self.data=self.data[self.data['label']!=-1]
        self.data=pd.DataFrame(self.data)

        return self.data

    def get_label(self):
        return self.label

            



    