import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
        self.x_train = self.x_test = self.y_train = self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data[[
            'person_age',
            'person_gender',
            'person_education',
            'person_income',
            'person_emp_exp',
            'person_home_ownership',
            'loan_amnt',
            'loan_intent',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length',
            'credit_score',
            'previous_loan_defaults_on_file']]

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_df, self.output_df, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test


class EncoderHandler:
    def __init__(self):
        self.ordinal_edu = None
        self.ordinal_home = None
        self.ohe_enc = None
        self.label_gender = None
        self.label_prev = None
        self.ordinal_cat_edu =  ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
        self.ordinal_cat_home =  ['OTHER', 'RENT', 'MORTGAGE', 'OWN']
        self.ohe_col = 'loan_intent'

    def fit_transform(self, df):
        df = df.copy()
        
        self.ordinal_edu = OrdinalEncoder(categories=[self.ordinal_cat_edu])
        self.ordinal_home = OrdinalEncoder(categories=[self.ordinal_cat_home])
        df['person_home_ownership'] = self.ordinal_home.fit_transform(df[['person_home_ownership']])
        df['person_education'] = self.ordinal_edu.fit_transform(df[['person_education']])
    
        self.ohe_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ohe_enc.fit(df[[self.ohe_col]])
        ohe_features = self.ohe_enc.transform(df[[self.ohe_col]])
        ohe_columns = self.ohe_enc.get_feature_names_out([self.ohe_col])
        df = df.drop(columns=[self.ohe_col])
        df[ohe_columns] = ohe_features.astype(int)
        
        self.label_gender = LabelEncoder()
        self.label_prev = LabelEncoder()
        df['person_gender'] = self.label_gender.fit_transform(df['person_gender'])
        df['previous_loan_defaults_on_file'] = self.label_prev.fit_transform(df['previous_loan_defaults_on_file'])

        return df

    def transform(self, df):
        df = df.copy()

        df['person_home_ownership'] = self.ordinal_home.transform(df[['person_home_ownership']])
        df['person_education'] = self.ordinal_edu.transform(df[['person_education']])

        ohe_features = self.ohe_enc.transform(df[[self.ohe_col]])
        ohe_columns = self.ohe_enc.get_feature_names_out([self.ohe_col])
        df = df.drop(columns=[self.ohe_col])
        df[ohe_columns] = ohe_features.astype(int)

        df['person_gender'] = self.label_gender.transform(df['person_gender'])
        df['previous_loan_defaults_on_file'] = self.label_prev.transform(df['previous_loan_defaults_on_file'])
        return df


class ScalerHandler:
    def __init__(self):
        self.scaler = RobustScaler()
        self.numeric_columns = [
            'person_age',
            'person_income',
            'person_emp_exp',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length',
            'credit_score']

    def fit_transform(self, df):
        df = df.copy()
        df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        return df

    def transform(self, df):
        df = df.copy()
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df


class ModelHandler:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        preds = self.model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        print("\n Model Accuracy:", acc)
        return acc

    def predict(self, x_test):
        return self.model.predict(x_test)

    def report(self, y_test, y_pred):
        print('\nClassification Report\n')
        print(classification_report(y_test, y_pred))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved as {filename}")


file_path = "Dataset_A_loan.csv"
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('loan_status')
x_train_raw, x_test_raw, y_train, y_test = data_handler.split_data()

encoder = EncoderHandler()
x_train_encoded = encoder.fit_transform(x_train_raw)
x_test_encoded = encoder.transform(x_test_raw)

with open("label_encoder_gender.pkl", 'wb') as f:
    pickle.dump(encoder.label_gender, f)

with open("label_encoder_prev.pkl", 'wb') as f:
    pickle.dump(encoder.label_prev, f)

with open("ordinal_encoder_education.pkl", 'wb') as f:
    pickle.dump(encoder.ordinal_edu, f)

with open("ordinal_encoder_home.pkl", 'wb') as f:
    pickle.dump(encoder.ordinal_home, f)

with open("ohe_encoder_loan_intent.pkl", 'wb') as f:
    pickle.dump(encoder.ohe_enc, f)


scaler = ScalerHandler()
x_train_scaled = scaler.fit_transform(x_train_encoded)
x_test_scaled = scaler.transform(x_test_encoded)

with open("scaler.pkl", 'wb') as f:
    pickle.dump(scaler.scaler, f)

model_handler = ModelHandler()
print("Training Model...")
model_handler.train(x_train_scaled, y_train)

accuracy = model_handler.evaluate(x_test_scaled, y_test)

y_pred = model_handler.predict(x_test_scaled)
model_handler.report(y_test, y_pred)

model_handler.save("xg_loan.pkl")

