for month in df.index.get_level_values(0):
    sub_df = df.loc[(month,)]
    marks = 'ABCDEFG'
    qxdict={
        '姓名':num,
        '月份':month
    }
    for mark in marks:
        if mark in sub_df.index:
            qxdict[mark]=sub_df.loc[mark]
    records.append(qxdict)
