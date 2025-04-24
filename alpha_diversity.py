import pandas as pd
import skbio.diversity.alpha as alpha_diversity
import numpy as np

class AlphaDiversityCalculator:
    def __init__(self, otu_table):
        self.otu_table = otu_table
        self.alpha_diversity_results = {}
        self.num_samples = len(otu_table.index)



    def calculate_alpha_diversity_metrics(self):

        #  Richness metrics
        #  ace, chao1, chao1_ci, doubles, faith_pd, margalef, menhinick, michaelis_menten_fit, observed_features, observed_otus, osd, singles, sobs 
        #  Diversity metrics
        #  
        self.alpha_diversity_results = {
            #'ace': [],
            'chao1': [],
            #'chao1_ci': [],  
            'berger': [],
            'brillouin': [],  
            'dominance': [], 
            'doubles': [],  
            'enspie': [],  
            #'esty_ci': [],  
            #'faith_pd': [],  # Placeholder: Requires phylogenetic tree
            'fisher': [],
            'gini': [],
            'goods': [],  
            'heip_e': [],  # Placeholder
            'hill': [],  # Placeholder: Requires calculation based on Hill numbers
            'inv_simpson': [],  # Use inverse Simpson’s index
            #'kempton_taylor_q': [],  # Placeholder: Rarely used, custom implementation needed
            #'lladser_pe': [],  # Placeholder: Custom metric
           # 'lladser_ci': [],  # Placeholder
            'margalef': [],  # Placeholder
            'mcintosh_d': [],  # Placeholder
            'mcintosh_e': [],  # Placeholder
            'menhinick': [],  # Placeholder
           # 'michaelis_menten_fit': [],  # Placeholder: Requires curve fitting
            # 'observed_features': [],  # Simply the OTU richness
            # 'observed_otus': [],  # Same as above, redundancy kept
            #'osd': [],  # Placeholder
            #'phydiv': [],  # Placeholder: Phylogenetic diversity
            'pielou_e': [],  # Placeholder: Evenness
            'renyi': [],  # Placeholder: Part of Renyi entropy
            'robbins': [],  # Placeholder
            'shannon': [],
            'simpson': [],
            'simpson_d': [],  # Simpson’s dominance
            'simpson_e': [],  # Simpson’s evenness
            'singles': [],  # Number of singletons (OTUs observed exactly once)
            'sobs': [],  # Synonym for observed OTUs
            'strong': [],  # Placeholder
            'tsallis': []  # Placeholder: Custom entropy calculation
        }
        for sample in self.otu_table.index:
            counts = self.otu_table.loc[sample].values
            # Calculate diversity metrics
            counts_integral=np.round(counts).astype(int)
           # self.alpha_diversity_results['ace'].append(alpha_diversity.ace(counts_integral))
            self.alpha_diversity_results['chao1'].append(alpha_diversity.chao1(counts))
           # self.alpha_diversity_results['chao1_ci'].append(alpha_diversity.chao1_ci(counts))  # reasons: its return value is an interval
            self.alpha_diversity_results['berger'].append(alpha_diversity.berger_parker_d(counts))
            self.alpha_diversity_results['brillouin'].append(alpha_diversity.brillouin_d(counts))
            self.alpha_diversity_results['dominance'].append(alpha_diversity.dominance(counts))
            self.alpha_diversity_results['doubles'].append(alpha_diversity.doubles(counts))
            self.alpha_diversity_results['enspie'].append(alpha_diversity.enspie(counts))
            #self.alpha_diversity_results['esty_ci'].append(alpha_diversity.esty_ci(counts))

            #self.alpha_diversity_results['faith_pd'].append(None) # placeholder:require phylogenetic tree
            self.alpha_diversity_results['fisher'].append(alpha_diversity.fisher_alpha(counts))
            self.alpha_diversity_results['gini'].append(alpha_diversity.gini_index(counts))# measure the inequality in species abundance
            self.alpha_diversity_results['goods'].append(alpha_diversity.goods_coverage(counts))
            self.alpha_diversity_results['heip_e'].append(alpha_diversity.heip_e(counts))
            self.alpha_diversity_results['hill'].append(alpha_diversity.hill(counts))
            self.alpha_diversity_results['inv_simpson'].append(alpha_diversity.inv_simpson(counts))
            # self.alpha_diversity_results['kempton_taylor_q'].append(alpha_diversity.kempton_taylor_q(counts)) # for some columns,just has one value
            #self.alpha_diversity_results['lladser_ci'].append(alpha_diversity.lladser_ci(counts))
            #self.alpha_diversity_results['lladser_pe'].append(alpha_diversity.lladser_pe(counts_integral))
            self.alpha_diversity_results['margalef'].append(alpha_diversity.margalef(counts))
            self.alpha_diversity_results['mcintosh_d'].append(alpha_diversity.mcintosh_d(counts))
            self.alpha_diversity_results['menhinick'].append(alpha_diversity.menhinick(counts))
            self.alpha_diversity_results['mcintosh_e'].append(alpha_diversity.mcintosh_e(counts))
            #self.alpha_diversity_results['michaelis_menten_fit'].append(alpha_diversity.michaelis_menten_fit(counts_integral))
            # self.alpha_diversity_results['observed_features'].append(alpha_diversity.observed_features(counts))
            # self.alpha_diversity_results['observed_otus'].append(alpha_diversity.observed_otus(counts))
            #self.alpha_diversity_results['osd'].append(alpha_diversity.osd(counts))
            #self.alpha_diversity_results['phydiv'].append(None)# require phylogenetic tree
            self.alpha_diversity_results['pielou_e'].append(alpha_diversity.pielou_e(counts))
            self.alpha_diversity_results['renyi'].append(alpha_diversity.renyi(counts))
            self.alpha_diversity_results['robbins'].append(alpha_diversity.robbins(counts))
            self.alpha_diversity_results['shannon'].append(alpha_diversity.shannon(counts))
            self.alpha_diversity_results['simpson'].append(alpha_diversity.simpson(counts))
            self.alpha_diversity_results['simpson_d'].append(alpha_diversity.simpson_d(counts))
            self.alpha_diversity_results['simpson_e'].append(alpha_diversity.simpson_e(counts))
            self.alpha_diversity_results['singles'].append(alpha_diversity.singles(counts))
            self.alpha_diversity_results['sobs'].append(alpha_diversity.sobs(counts))
            self.alpha_diversity_results['strong'].append(alpha_diversity.strong(counts))
            self.alpha_diversity_results['tsallis'].append(alpha_diversity.tsallis(counts))
            


            
           

    def get_results(self):
        '''
        Return the alpha diversity results.
        '''
        return pd.DataFrame(self.alpha_diversity_results, index=self.otu_table.index)



    