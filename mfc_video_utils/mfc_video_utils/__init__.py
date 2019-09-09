import pandas as pd
import skvideo.io
import fractions
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections.abc import MutableMapping
from pathlib import Path
from tqdm import tqdm
from IPython.display import HTML #For viewing videos / decision in the notebook
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, roc_auc_score, recall_score, accuracy_score, precision_score, f1_score, confusion_matrix, average_precision_score
import random


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


class MfcEvalVideoProcessor:
    ''' Class to process MFC video datasets easily'''
    frac_list = ['video_@codec_time_base', 'video_@r_frame_rate', 'video_@avg_frame_rate', 
             'video_@time_base', 'audio_@r_frame_rate','audio_@codec_time_base', 
             'audio_@time_base', 'subtitle_@r_frame_rate', 'subtitle_@time_base', 
             'data_@r_frame_rate', 'data_@avg_frame_rate', 'data_@time_base']
    time_list = ['video_tag_@creation_time', 'audio_tag_@creation_time', 'video_tag_@DateTime',
             'video_tag_@DateTimeOriginal', 'video_tag_@DateTimeDigitized', 'data_tag_@creation_time']
    
    def __init__(self, trained_clf: BaseEstimator):
        '''__init__ constructor'''
        self.classifier = trained_clf
    
    def __call__(self, path_to_video: str):
        return self.compute_video_score(path_to_video)
    
    def compute_video_score(self, path_to_video: str):
        processed_video = self._process_video(path_to_video)
        return self.classifier.predict_proba(processed_video)[:,1][0]
    
    def _process_video(self, path_to_video: str) -> pd.DataFrame:
        video_metadata = self._flatten(skvideo.io.ffprobe(path_to_video))
        video_df = pd.DataFrame.from_dict(video_metadata, orient='index')
        video_df = video_df.transpose()
        video_df = video_df.apply(pd.to_numeric, errors='ignore')
        for col in MfcEvalVideoProcessor.frac_list:
            if col in video_df.columns:
                video_df[col] = video_df[col].apply(self._conv_to_float)
        for col in MfcEvalVideoProcessor.time_list:
            def time_transform(x): 
                if pd.notnull(x) and type(x) is not str:
                    return x.to_datetime64().astype(np.int64) 
                else:
                    return np.nan
            if col in video_df.columns:
                video_df[col] = video_df[col].apply(pd.to_datetime, errors='ignore').apply(time_transform)
        return video_df
    
    def _flatten(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        items = []
        for k, v in d.items():
            new_key = '{0}{1}{2}'.format(parent_key,sep,k) if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # apply itself to each element of the list - that's it!
                dic_list = self._flatten_tag(v, parent_key=new_key)
                for k_2, v_2 in dic_list.items():
                    items.append((k_2, v_2))
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def _flatten_tag(l: list, parent_key: str = '', sep: str = '_') -> dict:
        items = []
        for ele in l:
            new_key = '{0}{1}{2}{3}'.format(parent_key,sep,'@',ele['@key']) 
            items.append((new_key, ele['@value']))
        return dict(items)

    @staticmethod
    def _conv_to_float(frac: str) -> float:
        if pd.notnull(frac):
            try: return float(fractions.Fraction(frac))
            except ZeroDivisionError: return 0
        else:
            return frac
    

class MfcVideoProcessor:
    ''' Class to process MFC video datasets easily'''
    frac_list = ['video_@codec_time_base', 'video_@r_frame_rate', 'video_@avg_frame_rate', 
             'video_@time_base', 'audio_@r_frame_rate','audio_@codec_time_base', 
             'audio_@time_base', 'subtitle_@r_frame_rate', 'subtitle_@time_base', 
             'data_@r_frame_rate', 'data_@avg_frame_rate', 'data_@time_base']
    time_list = ['video_tag_@creation_time', 'audio_tag_@creation_time', 'video_tag_@DateTime',
             'video_tag_@DateTimeOriginal', 'video_tag_@DateTimeDigitized', 'data_tag_@creation_time']
    
    def __init__(self, name: str, dataset_abs_path: Path, ref_avail: bool=False, ref_folder: bool=False):
        '''__init__ constructor'''
        if not dataset_abs_path.exists():
            raise Exception("Dataset path does not exist")
        else:
            self.name = name
            self.basepath = Path(dataset_abs_path)
            self.probes = self.basepath / "probe"
            self.ref_avail = ref_avail
            if ref_avail and ref_folder:
                self.reference_basepath = Path(str(dataset_abs_path) + '-Reference')
                if not self.reference_basepath.exists():
                    raise Exception("Path to reference folder does not exist")
                else:
                    self.csv_path = [item for item in (self.reference_basepath / "reference/manipulation-video/").glob("*-manipulation-video-ref.csv")][0]
                    self.csv = pd.read_csv(self.csv_path, sep="|")
                    self.labels = [self.get_video_label(i) for i in range(len(self.csv))]
            elif ref_avail:
                self.csv_path = [item for item in self.basepath.glob("reference/manipulation-video/*-manipulation-video-ref.csv")][0]
                self.csv = pd.read_csv(self.csv_path, sep="|")
                self.labels = [self.get_video_label(i) for i in range(len(self.csv))]
            else:
                self.csv_path = [item for item in self.basepath.glob("indexes/manipulation-video/*-manipulation-video-ref.csv")][0]
                self.csv = pd.read_csv(self.csv_path, sep="|")
            self.ffprobe_df = self._generate_ffprobe_dataset_dataframe()

    def get_video_path(self, video_id: int) -> str:
        video_name = str(self.csv.at[video_id, 'ProbeFileName'])
        return str(self.basepath / video_name)

    def get_video_label(self, video_id: int) -> int:
        if self.ref_avail:
            return 1 if self.csv.at[video_id, 'IsTarget'] is 'Y' else 0
        else:
            raise Exception("This dataset has no reference available")
            
    def play_video(self, vid_id: str) -> HTML:
        video_str = '"' +  self.get_video_path(vid_id) + '"'
        return HTML("""<video width="640" height="480" controls>
                 <source src=""" + video_str + """ >
                 </video>""")
            
    def _generate_ffprobe_dataset_dataframe(self) -> pd.DataFrame:
        dfs = [] #creates a new dataframe that's empty
        for i in tqdm(range(len(self.csv))):
            video_metadata = self._flatten(skvideo.io.ffprobe(self.get_video_path(i)))
            video_df = pd.DataFrame.from_dict(video_metadata, orient='index', columns=[i])
            dfs.append(video_df)
        vids_df = pd.concat(dfs, axis=1, sort=False)
        vids_df = vids_df.transpose()
        vids_df = vids_df.apply(pd.to_numeric, errors='ignore')
        for col in MfcVideoProcessor.frac_list:
            if col in vids_df.columns:
                vids_df[col] = vids_df[col].apply(self._conv_to_float)
        for col in MfcVideoProcessor.time_list:
            def time_transform(x): 
                if pd.notnull(x) and type(x) is not str:
                    return x.to_datetime64().astype(np.int64) 
                else:
                    return np.nan
            if col in vids_df.columns:
                vids_df[col] = vids_df[col].apply(pd.to_datetime, errors='ignore').apply(time_transform)
        return vids_df

    def _flatten(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        items = []
        for k, v in d.items():
            new_key = '{0}{1}{2}'.format(parent_key,sep,k) if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # apply itself to each element of the list - that's it!
                dic_list = self._flatten_tag(v, parent_key=new_key)
                for k_2, v_2 in dic_list.items():
                    items.append((k_2, v_2))
            else:
                items.append((new_key, v))
        return dict(items)

    def _flatten_tag(self, l: list, parent_key: str = '', sep: str = '_') -> dict:
        items = []
        for ele in l:
            new_key = '{0}{1}{2}{3}'.format(parent_key,sep,'@',ele['@key']) 
            items.append((new_key, ele['@value']))
        return dict(items)

    def _conv_to_float(self, frac: str) -> float:
        if pd.notnull(frac):
            try: return float(fractions.Fraction(frac))
            except ZeroDivisionError: return 0
        else:
            return frac
        

# ref: https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
class BasicTransformer(BaseEstimator):
    
    def __init__(self, cat_threshold=None, num_strategy='median', return_df=False):
        # store parameters as public attributes
        self.cat_threshold = cat_threshold
        
        if num_strategy not in ['mean', 'median']:
            raise ValueError('num_strategy must be either "mean" or "median"')
        self.num_strategy = num_strategy
        self.return_df = return_df
        
    def fit(self, X, y=None):
        # Assumes X is a DataFrame
        self._columns = X.columns.values
        
        # Split data into categorical and numeric
        self._dtypes = X.dtypes.values
        self._kinds = np.array([dt.kind for dt in X.dtypes])
        self._column_dtypes = {}
        is_cat = self._kinds == 'O'
        self._column_dtypes['cat'] = self._columns[is_cat]
        self._column_dtypes['num'] = self._columns[~is_cat]
        self._feature_names = self._column_dtypes['num']
        
        # Create a dictionary mapping categorical column to unique values above threshold
        self._cat_cols = {}
        for col in self._column_dtypes['cat']:
            vc = X[col].value_counts()
            if self.cat_threshold is not None:
                vc = vc[vc > self.cat_threshold]
            vals = vc.index.values
            self._cat_cols[col] = vals
            self._feature_names = np.append(self._feature_names, col + '_' + vals)
        
        # Compute mean and std for every numerical column
        self._num_cols_mean = {}
        self._num_cols_median = {}
        self._num_cols_std = {}
        for col in self._column_dtypes['num']:
            mean = X[col].fillna(0).values.mean()
            median = np.median(X[col].fillna(0).values)
            std = X[col].fillna(0).values.std()
            self._num_cols_mean[col] = mean
            self._num_cols_median[col] = median
            self._num_cols_std[col] = std


            
        # get total number of new categorical columns    
        self._total_cat_cols = sum([len(v) for col, v in self._cat_cols.items()])
        
        # get mean or median
        # self._num_fill = X[self._column_dtypes['num']].agg(self.num_strategy).fillna(0)
        return self
        
    def transform(self, X):
        # check that we have a DataFrame with same column names as the one we fit
#         if set(self._columns) != set(X.columns):
#             raise ValueError('Passed DataFrame has different columns than fit DataFrame')
#         elif len(self._columns) != len(X.columns):
#             raise ValueError('Passed DataFrame has different number of columns than fit DataFrame')
        X = X.reindex(columns = self._columns)
    
        # fill missing values    
        X_num = X[self._column_dtypes['num']].fillna(self._num_cols_mean if self.num_strategy == 'mean' else self._num_cols_median)

        
        # Standardize numerics
        #std = X_num.std()
        X_num = (X_num - pd.Series(self._num_cols_mean)) / pd.Series(self._num_cols_std)
        #zero_std = np.where(std == 0)[0]
        
        # If there is 0 standard deviation, then all values are the same. Set them to 0.
#         if len(zero_std) > 0:
#             X_num.iloc[:, zero_std] = 0
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(0).values
        
        # create separate array for new encoded categoricals
        X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        i = 0
        for col in self._column_dtypes['cat']:
            vals = self._cat_cols[col]
            for val in vals:
                X_cat[:, i] = X[col] == val
                i += 1
                
        # concatenate transformed numeric and categorical arrays
        data = np.column_stack((X_num, X_cat))
        
        # return either a DataFrame or an array
        if self.return_df:
            return pd.DataFrame(data=data, columns=self._feature_names)
        else:
            return data
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def get_feature_names(self):
        return self._feature_names
    

def compute_roc(X_test: np.ndarray, y_test: [], trained_clf: BaseEstimator, plot: bool = True, title: str = "") -> float:
    y_score = trained_clf.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)', size=8)
        plt.ylabel('True Positive Rate (Sensitivity)', size=8)
        plt.title('ROC Curve - ' + title, size=10)
        plt.legend(fontsize=7);
    
    return roc_auc

def grid_search_forest(X_train, y_train, X_test, y_test, proc_pipe, cv_strategy, refit_score='roc_auc_score', n_iter=1000):
    random_seed = 123
    np.random.seed(random_seed)
    random.seed(random_seed)

    param_grid = {
        'forest__min_samples_split': [1.0, 2, 5, 10, 15, 100], 
        'forest__min_samples_leaf': [1, 2, 5, 10],
        'forest__max_depth': [5, 9, 15, 25, 30, None], 
        'forest__n_estimators' : [120, 300, 500, 800, 1200],
        'forest__max_features': ['log2', 'sqrt', None],
        'transformer__cat_threshold': [None],
        'transformer__num_strategy': ['median']
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'f1_score': make_scorer(f1_score),
        'average_precision_score': make_scorer(average_precision_score)
    }


    grid_search = RandomizedSearchCV(proc_pipe, param_grid, scoring=scorers, refit=refit_score,
                               cv=cv_strategy, return_train_score=True, n_jobs=-1, n_iter=n_iter, random_state=random_seed, verbose=10)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    
    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], 
                 index=['neg', 'pos']))
    
    return grid_search

def grid_search_svm(X_train, y_train, X_test, y_test, proc_pipe, cv_strategy, refit_score='roc_auc_score', n_iter=1000):
    random_seed = 123
    np.random.seed(random_seed)
    random.seed(random_seed)

    param_grid = {
        'SVM__C': [0.001, 0.01, 1, 10, 100, 150, 200], 
        'SVM__gamma': ['auto', 'scale'],
        'SVM__class_weight': ['balanced', None],        
        'transformer__cat_threshold': [None],
        'transformer__num_strategy': ['median']
    }

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'f1_score': make_scorer(f1_score),
        'average_precision_score': make_scorer(average_precision_score)
    }


    grid_search = RandomizedSearchCV(proc_pipe, param_grid, scoring=scorers, refit=refit_score,
                               cv=cv_strategy, return_train_score=True, n_jobs=-1, n_iter=n_iter, random_state=random_seed, verbose=10)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)
    print(grid_search.best_score_)


    # confusion matrix on the test data.
    print('\nConfusion matrix of SVM optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], 
                 index=['neg', 'pos']))
    
    return grid_search