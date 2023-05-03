
import pandas as pd

# 读取数据
df = pd.read_excel('THSD(1).xlsx')

# 1. 插入新列'microclass'
df.insert(1, 'microclass', df['id'].apply(lambda x: 'Pro' if 'Pro' in x else 'Euk'))

# 2. 统计0值个数并删除符合条件的行
df = df[df.iloc[:, 2:].eq(0).sum(axis=1) <= 24]


# 3. 按'microclass'分组，计算每行值占组内各行数值加和的百分比
grouped = df.groupby('microclass')

df_r = pd.DataFrame(grouped.apply(lambda x: x.iloc[:, 2:].apply(lambda y: y / y.sum(), axis=0)).values.reshape(-1, df.shape[1]-2),columns=df.columns[2:])
print(df.shape,df_r.shape)
df.reset_index(drop=True, inplace=True)
df_r.reset_index(drop=True, inplace=True)
df = pd.concat([df, df_r], axis=1)
print(df.shape)

# 4. 根据'microclass'和其他条件删除符合条件的行
df1 = df[df['microclass'] == 'Pro'][~((df[df['microclass'] == 'Pro'].iloc[:, 32:]<0.001).sum(axis=1) == 30)]
print(df1.head())
df2 =  df[df['microclass'] == 'Euk'][~((df[df['microclass'] == 'Euk'].iloc[:, 32:]<0.01).sum(axis=1) == 30)]
df = pd.concat([df1, df2], axis=0)

# 5. 计算每行值的总和并添加到新列'sum_percent'
df['sum_percent'] = df.iloc[:, 32:].sum(axis=1)

# 6. 按'microclass'分组并以'sum_percent'降序排列
df = df.sort_values(['microclass', 'sum_percent'], ascending=[True, False]).groupby('microclass').apply(lambda x: x.reset_index(drop=True))

df = pd.concat([df.iloc[:,:32],df.iloc[:,-1]],axis=1)
# 7.导出处理后的Excel文件
df.to_excel('THSD_after.xlsx', index=False)







