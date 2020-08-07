def clean_data():
    df = pd.read_csv('../data/raw/train.csv')

    # delete unwanted features
    df = df.drop('ID',1)

    # missing values
    df['Ever_Married_NA'] = np.where(df['Ever_Married'].isnull(),1,0)
    df['Graduated_NA'] = np.where(df['Graduated'].isnull(),1,0)
    df['Profession_NA'] = np.where(df['Profession'].isnull(),1,0)
    df['Work_Experience_NA'] = np.where(df['Work_Experience'].isnull(),1,0)
    df['Family_Size_NA'] = np.where(df['Family_Size'].isnull(),1,0)
    df['Var_1_NA'] = np.where(df['Var_1'].isnull(),1,0)

    df.loc[(df['Ever_Married'].isna()) & (df['Age']<33), 'Ever_Married'] = 'No'
    df.loc[(df['Ever_Married'].isna()) & (df['Age']>=33), 'Ever_Married'] = 'Yes'
    df.loc[(df['Graduated'].isna()) & (df['Age'] < 39), 'Graduated'] = 'No'
    df.loc[(df['Graduated'].isna()) & (df['Age'] >= 39), 'Graduated'] = 'Yes'
    c = 'Profession'
    df[c] = df[c].fillna(df[c].mode()[0])
    c,grp = 'Work_Experience', 'Profession'
    df[c] = df[c].fillna(df.groupby(grp)[c].transform('mean').round())
    df['Age_cat'] = pd.qcut(df['Age'],q=3, labels=[0,1,2])
    df.loc[(df['Family_Size'].isna()) & (df['Age_cat'] == 0), 'Family_Size'] = 3.0
    df.loc[(df['Family_Size'].isna()) & (df['Age_cat'] != 0), 'Family_Size'] = 2.0
    df['Var_1'] = df['Var_1'].fillna(df['Var_1'].mode()[0])

    # numerical feature binning
    df['Family_Size_cat'] = pd.cut(df['Family_Size'],
                                   bins=[0,3,6,np.inf],
                                   labels=[0,1,2],
                                   include_lowest=True,
                                   right=False).to_numpy()
    df['Work_Experience_cat'] = pd.cut(df['Work_Experience'],
                                   bins=[0,3,7,np.inf],
                                   labels=[0,1,2],
                                   include_lowest=True,
                                   right=False).to_numpy()

    # categorical feature encoding
    df['Gender'] = df['Gender'].replace({'Male':0,'Female':1})
    df['Ever_Married'] = df['Ever_Married'].replace({'No':0,'Yes':1})
    df['Graduated'] = df['Graduated'].replace({'No':0,'Yes':1})

    df['Spending_Score'] = df['Spending_Score'].replace({'Low':0,'Average':1,
                                                        'High':2})

    df['Segmentation'] = df['Segmentation'].replace({'A':0,'B':1,'C':2,'D':3})
    df = pd.get_dummies(df, columns=['Profession','Var_1'], drop_first=False)

    # create cross features
    df['gen_mar']=df['Gender']+2*df['Ever_Married']
    df['gen_grad']=df['Gender']+2*df['Graduated']
    df['gen_spend']=df['Gender']+3*df['Spending_Score']
    df['grad_spend']=df['Graduated']+3*df['Spending_Score']
    df['grad_spend_gen']=df['Graduated']+3*df['Spending_Score']+9*df['Gender']
    
    # write clean data
    df.to_csv('../data/processed/clean_data.csv',index=False)
    
clean_data()
