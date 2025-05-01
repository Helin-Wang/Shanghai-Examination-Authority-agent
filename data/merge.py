import pandas as pd

def merge_df_row():
    new_df = pd.DataFrame()
    for start in range(0, 3500, 500):
        end = min(start + 500, 3241)
        #end = start + 500
        filepath = f"./考试院faq_triplets_different_question_{start}_{end}.csv"
        df = pd.read_csv(filepath).iloc[start:end]
        new_df = pd.concat([new_df, df])
        
    return new_df

def merge_df_column(df1, df2):
    if df1.shape[0] == df2.shape[0]:
        df = pd.concat([df1, df2], axis=1)
        return df

def add_index(df):
    df['index'] = df.index
    return df

def retriever_df_build(df):
    same_question_list = []
    same_question_source_list = []
    different_question_list = []
    different_question_source_list = []
    for index, row in df.iterrows():
        for same_question in eval(row['同类问题']):
            same_question_list.append(same_question)
            same_question_source_list.append(row['index'])
        
        for different_question in eval(row['不同问题']):
            different_question_list.append(different_question)
            different_question_source_list.append(row['index'])

    same_question_df = pd.DataFrame({'问题': same_question_list, '来源': same_question_source_list})
    different_question_df = pd.DataFrame({'问题': different_question_list, '来源': different_question_source_list})
    return same_question_df, different_question_df
  
new_df = merge_df_row()
print(new_df.shape)
print(new_df.columns)
new_df.to_csv("./考试院faq_triplets_different_question_重写.csv", index=False)