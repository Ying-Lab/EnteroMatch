from skbio.diversity import beta_diversity
from scipy.stats import zscore
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

class BetaDiversityCalculus:
    def __init__(self,donor_otu,recipient_otu,data):
        self.donor=donor_otu
        self.recipient=recipient_otu
        self.data=data

    def calculate_beta_diversity(self):
    
        bray_curtis_matrix = []
        jaccard_matrix=[]
        euclidean_matrix=[]
        cityblock_matrix=[]
        canberra_matrix=[]
        chebyshev_matrix=[]
        correlation_matrix=[]
        cosine_matrix=[]
        dice_matrix=[]
        hamming_matrix=[]
        # mahalanobis_matrix=[]
        manhattan_matrix=[]
        matching_matrix=[]
        minkowski_matrix=[]
        rogerstanimoto_matrix=[]
        russellrao_matrix=[]
        # seuclidean_matrix=[]
        sokalmichener_matrix=[]
        sokalsneath_matrix=[]
        sqeuclidean_matrix=[]
        yule_matrix=[]

        # 遍历每个 donor 样本
        for donor_index in range(self.donor.shape[0]):
            # 提取当前的 donor 样本
            selected_donor = self.donor.iloc[donor_index, :].values

            # 计算该 donor 样本与所有 recipient 样本之间的 Bray-Curtis dissimilarity
            donor_vs_recipient_bray = np.array([beta_diversity('braycurtis', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            
            donor_vs_recipient_jaccard = np.array([beta_diversity('jaccard', 
                                                                  np.vstack([selected_donor, recipient]), 
                                                                  ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                  for i, recipient in enumerate(self.recipient.values)])
            
            donor_vs_recipient_euclidean= np.array([beta_diversity('euclidean', 
                                                                   np.vstack([selected_donor, recipient]), 
                                                                   ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                   for i, recipient in enumerate(self.recipient.values)])
            
            donor_vs_recipient_cityblock=np.array([beta_diversity('cityblock', 
                                                                   np.vstack([selected_donor, recipient]), 
                                                                   ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                   for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_canberra=np.array([beta_diversity('canberra', 
                                                                   np.vstack([selected_donor, recipient]), 
                                                                   ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                   for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_chebyshev=np.array([beta_diversity('chebyshev', 
                                                                   np.vstack([selected_donor, recipient]), 
                                                                   ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                   for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_correlation=np.array([beta_diversity('correlation', 
                                                                   np.vstack([selected_donor, recipient]), 
                                                                   ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                                   for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_cosine = np.array([beta_diversity('cosine', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_dice = np.array([beta_diversity('dice', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            
            donor_vs_recipient_hamming = np.array([beta_diversity('hamming', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            # # donor_vs_recipient_mahalanobis=np.array([beta_diversity('mahalanobis', 
            #                                                    np.vstack([selected_donor, recipient]), 
            #                                                    ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
            #                                                    for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_mahattan=np.array([beta_diversity('manhattan', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_matching=np.array([beta_diversity('matching', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_minkowski=np.array([beta_diversity('minkowski', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_rogerstanimoto=np.array([beta_diversity('rogerstanimoto', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_russellrao=np.array([beta_diversity('russellrao', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            # donor_vs_recipient_seuclidean=np.array([beta_diversity('seuclidean', 
            #                                                    np.vstack([selected_donor, recipient]), 
            #                                                    ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
            #                                                    for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_sokalmichener=np.array([beta_diversity('sokalmichener', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_sokalsneath=np.array([beta_diversity('sokalsneath', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_sqeuclidean=np.array([beta_diversity('sqeuclidean', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            donor_vs_recipient_yule=np.array([beta_diversity('yule', 
                                                               np.vstack([selected_donor, recipient]), 
                                                               ids=[f"donor_{donor_index}", f"recipient_{i}"])[0, 1] 
                                                               for i, recipient in enumerate(self.recipient.values)])
            
            # 提取当前 donor 的 dissimilarity 第一行
            bray_curtis_matrix.append(donor_vs_recipient_bray)
            jaccard_matrix.append(donor_vs_recipient_jaccard)
            euclidean_matrix.append(donor_vs_recipient_euclidean)
            cityblock_matrix.append(donor_vs_recipient_cityblock)
            canberra_matrix.append(donor_vs_recipient_canberra)
            chebyshev_matrix.append(donor_vs_recipient_chebyshev)
            correlation_matrix.append(donor_vs_recipient_correlation)
            cosine_matrix.append(donor_vs_recipient_cosine)
            dice_matrix.append(donor_vs_recipient_dice)
            hamming_matrix.append(donor_vs_recipient_hamming)
            # mahalanobis_matrix.append(donor_vs_recipient_mahalanobis)
            manhattan_matrix.append(donor_vs_recipient_mahattan)
            matching_matrix.append(donor_vs_recipient_matching)
            minkowski_matrix.append(donor_vs_recipient_minkowski)
            rogerstanimoto_matrix.append(donor_vs_recipient_rogerstanimoto)
            russellrao_matrix.append(donor_vs_recipient_russellrao)
            # seuclidean_matrix.append(donor_vs_recipient_seuclidean)
            sokalmichener_matrix.append(donor_vs_recipient_sokalmichener)
            sokalsneath_matrix.append(donor_vs_recipient_sokalsneath)
            sqeuclidean_matrix.append(donor_vs_recipient_sqeuclidean)
            yule_matrix.append(donor_vs_recipient_yule)

        # 将所有 donor 样本与 recipient 样本的 dissimilarities 转换为 DataFrame
        bray_curtis_matrix = pd.DataFrame(bray_curtis_matrix, index=self.donor.index, columns=self.recipient.index)
        jaccard_matrix=pd.DataFrame(jaccard_matrix,index=self.donor.index,columns=self.recipient.index)
        euclidean_matrix=pd.DataFrame(euclidean_matrix,index=self.donor.index,columns=self.recipient.index)
        cityblock_matrix=pd.DataFrame(cityblock_matrix,index=self.donor.index,columns=self.recipient.index)
        canberra_matrix=pd.DataFrame(canberra_matrix,index=self.donor.index,columns=self.recipient.index)
        chebyshev_matrix=pd.DataFrame(chebyshev_matrix,index=self.donor.index,columns=self.recipient.index)
        correlation_matrix=pd.DataFrame(correlation_matrix,index=self.donor.index,columns=self.recipient.index)
        cosine_matrix=pd.DataFrame(cosine_matrix,index=self.donor.index,columns=self.recipient.index)
        dice_matrix=pd.DataFrame(dice_matrix,index=self.donor.index,columns=self.recipient.index)
        hamming_matrix=pd.DataFrame(hamming_matrix,index=self.donor.index,columns=self.recipient.index)
        # mahalanobis_matrix=pd.DataFrame(mahalanobis_matrix,index=self.donor.index,columns=self.recipient.index)
        manhattan_matrix=pd.DataFrame(manhattan_matrix,index=self.donor.index,columns=self.recipient.index)
        matching_matrix=pd.DataFrame(matching_matrix,index=self.donor.index,columns=self.recipient.index)
        minkowski_matrix=pd.DataFrame(minkowski_matrix,index=self.donor.index,columns=self.recipient.index)
        rogerstanimoto_matrix=pd.DataFrame(rogerstanimoto_matrix,index=self.donor.index,columns=self.recipient.index)
        russellrao_matrix=pd.DataFrame(russellrao_matrix,index=self.donor.index,columns=self.recipient.index)
        # seuclidean_matrix=pd.DataFrame(seuclidean_matrix,index=self.donor.index,columns=self.recipient.index)
        sokalmichener_matrix=pd.DataFrame(sokalmichener_matrix,index=self.donor.index,columns=self.recipient.index)
        sokalsneath_matrix=pd.DataFrame(sokalsneath_matrix,index=self.donor.index,columns=self.recipient.index)
        sqeuclidean_matrix=pd.DataFrame(sqeuclidean_matrix,index=self.donor.index,columns=self.recipient.index)
        yule_matrix=pd.DataFrame(yule_matrix,index=self.donor.index,columns=self.recipient.index)

        self.modify_matrix(euclidean_matrix,'euclidean')
        self.modify_matrix(bray_curtis_matrix,'bray curtis')
        self.modify_matrix(jaccard_matrix,'jaccard')
        self.modify_matrix(cityblock_matrix,'cityblock')
        self.modify_matrix(canberra_matrix,'canberra')
        self.modify_matrix(chebyshev_matrix,'chebyshev')
        self.modify_matrix(correlation_matrix,'correlation')
        self.modify_matrix(cosine_matrix,'cosine')
        self.modify_matrix(dice_matrix,'dice')
        self.modify_matrix(hamming_matrix,'hamming')
        # self.modify_matrix(mahalanobis_matrix,'mahalanobis')
        self.modify_matrix(manhattan_matrix,'manhattan')
        self.modify_matrix(matching_matrix,'matching')
        self.modify_matrix(minkowski_matrix,'minkowski')
        self.modify_matrix(rogerstanimoto_matrix,'rogerstanimoto')
        self.modify_matrix(russellrao_matrix,'russellrao')
        # self.modify_matrix(seuclidean_matrix,'seuclidean')
        self.modify_matrix(sokalmichener_matrix,'sokalmichener')
        self.modify_matrix(sokalsneath_matrix,'sokalsneath')
        self.modify_matrix(sqeuclidean_matrix,'sqeuclidean')
        self.modify_matrix(yule_matrix,'yule')


        # stack_bray=bray_curtis_matrix.stack().reset_index()
        # stack_bray.columns=['donor','recipient','bray curtis']
        # self.data = pd.merge(self.data, stack_bray, on=['donor', 'recipient'], how='left')
        # self.data.insert(self.data.columns.get_loc('recipient tsallis')+1,'bray curtis',self.data.pop('bray curtis'))

    def modify_matrix(self,matrix,title):
        stack=matrix.stack().reset_index()

        # Standardize beta diversity
        stack.columns=['donor','recipient',title]
        stack[title]=zscore(stack[title])

        self.data = pd.merge(self.data, stack, on=['donor', 'recipient'], how='left')
        self.data.insert(self.data.columns.get_loc('recipient tsallis')+1,title,self.data.pop(title))

    def getData(self):
        return self.data
    


        

            