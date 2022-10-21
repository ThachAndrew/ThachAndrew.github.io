---
title: Collective Graph Identification
nav-title: Collective Graph Identification
category: projects
layout: default
---

*In progress*

Here, we solve several canonical problems in network analysis: entity resolution (determining when two observations correspond to the same entity), link prediction (inferring the existence of links), and node labeling (inferring hidden attributes).

Updates can be seen here:
- <https://github.com/linqs/collective-graph-identification-refactor>
- <https://github.com/ThachAndrew/psl-examples/tree/collective-graph-identification/collective-graph-identification>

# This notebook stores each step of refactoring the graph data into PSL data


```python
import pandas as pd
import re
import itertools # for cross products when filling in a full PSL dataset
```

## These functions help parse the .tab files.


```python
# assigns types to each column
def resolve_column_type(table):
    for column in table.columns:
        if column in {'id', 'email', 'alt_email', 'numsent', 'numreceived', 'numexchanged'}:
            table[column] = table[column].astype(str).astype(float).astype(int)
        # convert bag-of-words columns to floats (since ints won't take NaNs)
        elif re.match("w-", column):
            table[column] = table[column].astype(str).astype(float)

# extracts feature name from an element in a raw tab row
# returns: tuple (feature_name, feature_value, optional_value)
def get_feature_tuple(feature):
    feature_data = re.split(r"[:=]", feature)
    return feature_data
    

# loads the *.tab files into a Pandas Dataframe.
# returns: pd.DataFrame(columns=features)
def load_table(filename):

    # initialize the pandas dataframe
    node_data = pd.DataFrame()


    with open(filename) as infile:
        i = 0
        row_list = []
        for row in infile:
    
            #print('i is: ', i)

            if i == 0:
                # Skip non-useful first line
                print("Header: ", row)
            elif i == 1:
                # Prepare dataframe column labels
                tokens = row.split()
                if len(tokens) == 1:
                    print("This is not a NODE file, so don't load this row")
                else:  
                    features = ["id"] + [get_feature_tuple(feature)[1] for feature in tokens]
                    node_data = pd.DataFrame(columns=features)
            else:
          
                # this is to help the function generalize among the NODE and EDGE files.
                # EDGE files have a "|" character, which needs to be removed for proper feature decoupling
                row = re.sub(r'\|','', row)
            
                tokens = row.split()

                # the first token doesn't need splitting
                row_dict = {'id':tokens[0]}
                row_dict.update({get_feature_tuple(token)[0]:get_feature_tuple(token)[1] for token in tokens[1:]})
                row_list.append(row_dict)
        
            i += 1
        
        # Fill in rows
        node_data = pd.concat([node_data, pd.DataFrame(row_list)], ignore_index=True)

    return node_data
```

# Process the email nodes


```python
email_nodes = load_table('../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-1of6/sample-enron.NODE.email.tab')
# remove the (unnecessary) second to last column (it came from an ambiguous parse splits)
email_nodes.drop('other,manager,specialist,director,executive', axis=1, inplace=True)
resolve_column_type(email_nodes)

email_nodes.dtypes
```

    Header:  NODE	email
    





    id                int64
    emailaddress     object
    numsent           int64
    numreceived       int64
    numexchanged      int64
                     ...   
    w-kinney        float64
    w-veselack      float64
    w-mwhitt        float64
    w-jarnold       float64
    title            object
    Length: 5119, dtype: object




```python
email_nodes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>emailaddress</th>
      <th>numsent</th>
      <th>numreceived</th>
      <th>numexchanged</th>
      <th>w-gerald</th>
      <th>w-know</th>
      <th>w-busi</th>
      <th>w-mexicana</th>
      <th>w-transact</th>
      <th>...</th>
      <th>w-bartlo</th>
      <th>w-columbiagassubject</th>
      <th>w-perron</th>
      <th>w-coh</th>
      <th>w-agl</th>
      <th>w-kinney</th>
      <th>w-veselack</th>
      <th>w-mwhitt</th>
      <th>w-jarnold</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98</td>
      <td>scott.goodell@enron.com</td>
      <td>98</td>
      <td>607</td>
      <td>705</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>specialist</td>
    </tr>
    <tr>
      <th>1</th>
      <td>283</td>
      <td>c..koehler@enron.com</td>
      <td>128</td>
      <td>606</td>
      <td>734</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>p..south@enron.com</td>
      <td>8</td>
      <td>351</td>
      <td>359</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>3</th>
      <td>204</td>
      <td>lavorato@enron.com</td>
      <td>388</td>
      <td>3</td>
      <td>391</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>executive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>303</td>
      <td>t..hodge@enron.com</td>
      <td>95</td>
      <td>570</td>
      <td>665</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>206</th>
      <td>114</td>
      <td>vkamins@enron.com</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>207</th>
      <td>282</td>
      <td>sean.crandall@enron.com</td>
      <td>94</td>
      <td>138</td>
      <td>232</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>208</th>
      <td>270</td>
      <td>david.duran@enron.com</td>
      <td>7</td>
      <td>145</td>
      <td>152</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>209</th>
      <td>243</td>
      <td>kevin.presto@enron.com</td>
      <td>248</td>
      <td>198</td>
      <td>446</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>executive</td>
    </tr>
    <tr>
      <th>210</th>
      <td>131</td>
      <td>dave.fuller@enron.com</td>
      <td>165</td>
      <td>129</td>
      <td>294</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>211 rows × 5119 columns</p>
</div>



# Process the CoRef edges


```python
# need to rename one of the columns due to key collision
# use copy for safety
!cp ../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-1of6/sample-enron.UNDIRECTED.coref.tab .
!sed -i 's/email/alt_email/2g' sample-enron.UNDIRECTED.coref.tab

coref_edges = load_table('sample-enron.UNDIRECTED.coref.tab')
resolve_column_type(coref_edges)

coref_edges.dtypes
```

    Header:  UNDIRECTED	coref
    
    This is not a NODE file, so don't load this row





    id            int64
    email         int64
    alt_email     int64
    exists       object
    dtype: object




```python
coref_edges
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>email</th>
      <th>alt_email</th>
      <th>exists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2856</td>
      <td>265</td>
      <td>141</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18491</td>
      <td>310</td>
      <td>295</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>2</th>
      <td>516</td>
      <td>272</td>
      <td>183</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5131</td>
      <td>201</td>
      <td>19</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12417</td>
      <td>138</td>
      <td>78</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20776</th>
      <td>15003</td>
      <td>208</td>
      <td>135</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>20777</th>
      <td>4450</td>
      <td>197</td>
      <td>47</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>20778</th>
      <td>20302</td>
      <td>25</td>
      <td>248</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>20779</th>
      <td>12985</td>
      <td>222</td>
      <td>118</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20780</th>
      <td>19684</td>
      <td>248</td>
      <td>54</td>
      <td>NOTEXIST</td>
    </tr>
  </tbody>
</table>
<p>20781 rows × 4 columns</p>
</div>




```python
# Sanity Check: These should print pairs of the same people
for index in coref_edges[coref_edges['exists'] == 'EXIST'][['email', 'alt_email']].index:
    email_id  = coref_edges.loc[index]['email']
    alt_email_id = coref_edges.loc[index]['alt_email']

    print(email_nodes[email_nodes['id'] == email_id]['emailaddress'])
    print(email_nodes[email_nodes['id'] == alt_email_id]['emailaddress'])
    print("------------------------------------------------")
    
```

    206    vkamins@enron.com
    Name: emailaddress, dtype: object
    110    j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    21    d..baughman@enron.com
    Name: emailaddress, dtype: object
    77    don.baughman@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    209    kevin.presto@enron.com
    Name: emailaddress, dtype: object
    141    kpresto@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    36    m..tholt@enron.com
    Name: emailaddress, dtype: object
    43    jane.tholt@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    182    t..lucci@enron.com
    Name: emailaddress, dtype: object
    16    paul.lucci@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    85    mwhitt@ect.enron.com
    Name: emailaddress, dtype: object
    136    mark.whitt@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    39    bill.iii@enron.com
    Name: emailaddress, dtype: object
    60    gwendolyn.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    122    haedicke.mark@enron.com
    Name: emailaddress, dtype: object
    40    mark.e.haedicke@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    47    tmartin@enron.com
    Name: emailaddress, dtype: object
    69    thomas.martin@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    110    j.kaminski@enron.com
    Name: emailaddress, dtype: object
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    160    ryan.williams@enron.com
    Name: emailaddress, dtype: object
    195    bill.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    168    j..sturm@enron.com
    Name: emailaddress, dtype: object
    6    fletcher.sturm@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    52    lcampbel@enron.com
    Name: emailaddress, dtype: object
    13    larry.f.campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    151    vkamins@ect.enron.com
    Name: emailaddress, dtype: object
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    204    gsolberg@enron.com
    Name: emailaddress, dtype: object
    59    geir.solberg@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    85    mwhitt@ect.enron.com
    Name: emailaddress, dtype: object
    150    mwhitt@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    122    haedicke.mark@enron.com
    Name: emailaddress, dtype: object
    62    e..haedicke@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    187    j..kaminski@enron.com
    Name: emailaddress, dtype: object
    148    kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    80    tori.kuykendall@enron.com
    Name: emailaddress, dtype: object
    139    tkuyken@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    17    barbara.gray@enron.com
    Name: emailaddress, dtype: object
    158    n..gray@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    49    randall.gay@enron.com
    Name: emailaddress, dtype: object
    169    l..gay@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    195    bill.williams@enron.com
    Name: emailaddress, dtype: object
    188    c..williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    161    sscott5@enron.com
    Name: emailaddress, dtype: object
    170    m..scott@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    131    scott.neal@enron.com
    Name: emailaddress, dtype: object
    99    sneal@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    175    kholst@enron.com
    Name: emailaddress, dtype: object
    24    keith.holst@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    34    rogers.herndon@enron.com
    Name: emailaddress, dtype: object
    9    herndon@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    3    lavorato@enron.com
    Name: emailaddress, dtype: object
    68    john.ex@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    101    m..forney@enron.com
    Name: emailaddress, dtype: object
    67    john.forney@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    206    vkamins@enron.com
    Name: emailaddress, dtype: object
    151    vkamins@ect.enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    127    susan.m.scott@enron.com
    Name: emailaddress, dtype: object
    170    m..scott@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    197    m..presto@enron.com
    Name: emailaddress, dtype: object
    141    kpresto@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    87    f..keavey@enron.com
    Name: emailaddress, dtype: object
    180    peter.f.keavey@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    116    kay.mann@enron.com
    Name: emailaddress, dtype: object
    203    kmann@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    62    e..haedicke@enron.com
    Name: emailaddress, dtype: object
    92    mark.haedicke@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    78    phillip.allen@enron.com
    Name: emailaddress, dtype: object
    38    pallen@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    82    e..dickson@enron.com
    Name: emailaddress, dtype: object
    58    stacy.dickson@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    147    f..brawner@enron.com
    Name: emailaddress, dtype: object
    163    sandra.brawner@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    52    lcampbel@enron.com
    Name: emailaddress, dtype: object
    126    larry.campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    122    haedicke.mark@enron.com
    Name: emailaddress, dtype: object
    92    mark.haedicke@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    174    a.taylor@enron.com
    Name: emailaddress, dtype: object
    83    mark.taylor@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    52    lcampbel@enron.com
    Name: emailaddress, dtype: object
    156    f..campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    62    e..haedicke@enron.com
    Name: emailaddress, dtype: object
    40    mark.e.haedicke@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    60    gwendolyn.williams@enron.com
    Name: emailaddress, dtype: object
    195    bill.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    198    mcuilla@enron.com
    Name: emailaddress, dtype: object
    53    martin.cuilla@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    56    vince.kaminski@enron.com
    Name: emailaddress, dtype: object
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    110    j.kaminski@enron.com
    Name: emailaddress, dtype: object
    187    j..kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    110    j.kaminski@enron.com
    Name: emailaddress, dtype: object
    151    vkamins@ect.enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    161    sscott5@enron.com
    Name: emailaddress, dtype: object
    140    sscott3@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    110    j.kaminski@enron.com
    Name: emailaddress, dtype: object
    148    kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    61    k..allen@enron.com
    Name: emailaddress, dtype: object
    38    pallen@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    47    tmartin@enron.com
    Name: emailaddress, dtype: object
    30    a..martin@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    109    w..pereira@enron.com
    Name: emailaddress, dtype: object
    76    susan.pereira@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    8    e..williams@enron.com
    Name: emailaddress, dtype: object
    39    bill.iii@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    8    e..williams@enron.com
    Name: emailaddress, dtype: object
    195    bill.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    206    vkamins@enron.com
    Name: emailaddress, dtype: object
    187    j..kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    64    j..farmer@enron.com
    Name: emailaddress, dtype: object
    19    daren.farmer@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    74    h..lewis@enron.com
    Name: emailaddress, dtype: object
    25    andrew.lewis@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    37    gstorey@enron.com
    Name: emailaddress, dtype: object
    186    geoff.storey@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    160    ryan.williams@enron.com
    Name: emailaddress, dtype: object
    188    c..williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    161    sscott5@enron.com
    Name: emailaddress, dtype: object
    127    susan.m.scott@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    206    vkamins@enron.com
    Name: emailaddress, dtype: object
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    27    paul.thomas@enron.com
    Name: emailaddress, dtype: object
    118    d..thomas@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    152    dperlin@enron.com
    Name: emailaddress, dtype: object
    35    debra.perlingiere@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    66    steven.south@enron.com
    Name: emailaddress, dtype: object
    2    p..south@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    42    john.lavorato@enron.com
    Name: emailaddress, dtype: object
    3    lavorato@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    70    s..shively@enron.com
    Name: emailaddress, dtype: object
    177    hunter.shively@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    170    m..scott@enron.com
    Name: emailaddress, dtype: object
    146    susan.scott@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    87    f..keavey@enron.com
    Name: emailaddress, dtype: object
    162    peter.keavey@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    151    vkamins@ect.enron.com
    Name: emailaddress, dtype: object
    56    vince.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    127    susan.m.scott@enron.com
    Name: emailaddress, dtype: object
    140    sscott3@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    69    thomas.martin@enron.com
    Name: emailaddress, dtype: object
    30    a..martin@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    126    larry.campbell@enron.com
    Name: emailaddress, dtype: object
    156    f..campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    12    w.duran@enron.com
    Name: emailaddress, dtype: object
    208    david.duran@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    39    bill.iii@enron.com
    Name: emailaddress, dtype: object
    195    bill.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    148    kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    126    larry.campbell@enron.com
    Name: emailaddress, dtype: object
    13    larry.f.campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    156    f..campbell@enron.com
    Name: emailaddress, dtype: object
    13    larry.f.campbell@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    86    brad.mckay@enron.com
    Name: emailaddress, dtype: object
    48    bmckay@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    32    b..sanders@enron.com
    Name: emailaddress, dtype: object
    123    richard.sanders@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    4    t..hodge@enron.com
    Name: emailaddress, dtype: object
    179    jeffrey.hodge@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    180    peter.f.keavey@enron.com
    Name: emailaddress, dtype: object
    162    peter.keavey@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    206    vkamins@enron.com
    Name: emailaddress, dtype: object
    148    kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    187    j..kaminski@enron.com
    Name: emailaddress, dtype: object
    81    vince.j.kaminski@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    197    m..presto@enron.com
    Name: emailaddress, dtype: object
    209    kevin.presto@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------
    8    e..williams@enron.com
    Name: emailaddress, dtype: object
    160    ryan.williams@enron.com
    Name: emailaddress, dtype: object
    ------------------------------------------------


# Process the Manager edges


```python
# Load in the email-submgr and sanity check the edges to see who is the manager of whom.
# need to rename one of the columns due to key collision
# use copy for safety
!cp ../c3/namata-kdd11-data/enron/enron-samples-lowunk/enron-sample-lowunk-1of6/sample-enron.UNDIRECTED.email-submgr.tab .
# FIXME: this is tainting the column names
!sed -i 's/\temail/\talt_email/2g' sample-enron.UNDIRECTED.email-submgr.tab

manager_edges = load_table('sample-enron.UNDIRECTED.email-submgr.tab')

# FIXME: can probably omit this line
manager_edges.drop('NOTEXIST,EXIST', axis=1, inplace=True)

resolve_column_type(manager_edges)

manager_edges.dtypes
```

    Header:  UNDIRECTED	email-submgr
    





    id                int64
    w-gerald        float64
    w-know          float64
    w-busi          float64
    w-mexicana      float64
                     ...   
    w-jarnold       float64
    numexchanged      int64
    email             int64
    alt_email         int64
    exists           object
    Length: 5118, dtype: object




```python
manager_edges
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>w-gerald</th>
      <th>w-know</th>
      <th>w-busi</th>
      <th>w-mexicana</th>
      <th>w-transact</th>
      <th>w-want</th>
      <th>w-thing</th>
      <th>w-review</th>
      <th>w-questar</th>
      <th>...</th>
      <th>w-coh</th>
      <th>w-agl</th>
      <th>w-kinney</th>
      <th>w-veselack</th>
      <th>w-mwhitt</th>
      <th>w-jarnold</th>
      <th>numexchanged</th>
      <th>email</th>
      <th>alt_email</th>
      <th>exists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2693</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>286</td>
      <td>324</td>
      <td>EXIST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2634</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>37</td>
      <td>74</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1256</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>148</td>
      <td>131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>313</td>
      <td>57</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1344</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>24</td>
      <td>170</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2046</th>
      <td>2105</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>67</td>
      <td>288</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>2047</th>
      <td>2374</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>237</td>
      <td>198</td>
      <td>212</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>3464</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>160</td>
      <td>210</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>2049</th>
      <td>531</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>316</td>
      <td>188</td>
      <td>NOTEXIST</td>
    </tr>
    <tr>
      <th>2050</th>
      <td>2026</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>273</td>
      <td>34</td>
      <td>NOTEXIST</td>
    </tr>
  </tbody>
</table>
<p>2051 rows × 5118 columns</p>
</div>



# Split data into observed and targets (AKA train and test)


```python
email_nodes_observed = email_nodes[email_nodes['title'].notna()]
email_nodes_targets = email_nodes[email_nodes['title'].isna()]

coref_edges_observed = coref_edges[coref_edges['exists'].notna()]
coref_edges_targets = coref_edges[coref_edges['exists'].isna()]

manager_edges_observed = manager_edges[manager_edges['exists'].notna()]
manager_edges_targets = manager_edges[manager_edges['exists'].isna()]
```


```python
# Sanity check to see if the splits match up with the paper.

print("email_node_observed: ", len(email_nodes_observed))
print("email_node_target: ", len(email_nodes_targets))

print("coref_edges_observed: ", len(coref_edges_observed))
print("coref_edges_target: ", len(coref_edges_targets))

print("manager_edges_observed: ", len(manager_edges_observed))
print("manager_edges_target: ", len(manager_edges_targets))
```

    email_node_observed:  171
    email_node_target:  40
    coref_edges_observed:  16625
    coref_edges_target:  4156
    manager_edges_observed:  1642
    manager_edges_target:  409


# Prepare data for PSL predicates


```python
# Takes a table and fills the missing pairs and values to specify a full, sufficient set
# So far it only works with binary predicates
def fill_observed_missing_possibilities(table, arguments, values):
    total_possibilities = set(itertools.product(list(table[arguments[0]]), values))
    already_observed_possibilities = set((table.loc[index][arguments[0]], table.loc[index][arguments[1]]) for index in table.index)

    missing_possibilities = total_possibilities - already_observed_possibilities
    row_list = []
    for arg_0, arg_1 in missing_possibilities:
        row_dict = {arguments[0]:arg_0, arguments[1]:arg_1, arguments[2]:0 }
        row_list.append(row_dict)
        
    return pd.concat([table, pd.DataFrame(row_list)])
```

## Predicate: EmailHasLabel(E, L)

### Observed


```python
title_map = {"other": 0, "manager": 1, "specialist": 2, "director": 3, "executive": 4}

# The copy is to suppress an in-place warning
email_has_label_obs = email_nodes_observed[['id', 'title']].copy()
# convert titles to integers, so PSL can ground faster
email_has_label_obs['title'] = email_has_label_obs['title'].map(title_map)

# add in an existence column
email_has_label_obs['exists'] = 1.0
```


```python
# Specify the full observed set, add in 1s for the observed, and 0s for the missing possibilities
full_set_email_has_label_obs = fill_observed_missing_possibilities(email_has_label_obs, ['id', 'title', 'exists'], list(title_map.values()))
full_set_email_has_label_obs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>exists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>283</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>204</td>
      <td>4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>318</td>
      <td>4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>679</th>
      <td>308</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>680</th>
      <td>57</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>681</th>
      <td>202</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>682</th>
      <td>236</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>683</th>
      <td>26</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>855 rows × 3 columns</p>
</div>




```python
# Outputs to file
# full_set_email_has_label_obs.to_csv('EmailHasLabel_obs.csv', sep ='\t', index=False, header=False)
```

### Truth/Targets


```python
ground_truth_email_nodes = load_table('../c3/namata-kdd11-data/enron/enron-samples-lowunk/outputgraph/enron.NODE.email.tab')
resolve_column_type(ground_truth_email_nodes)
ground_truth_email_nodes
```

    Header:  NODE	email
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>emailaddress</th>
      <th>numsent</th>
      <th>numreceived</th>
      <th>numexchanged</th>
      <th>w-gerald</th>
      <th>w-know</th>
      <th>w-busi</th>
      <th>w-mexicana</th>
      <th>w-transact</th>
      <th>...</th>
      <th>w-columbiagassubject</th>
      <th>w-perron</th>
      <th>w-coh</th>
      <th>w-agl</th>
      <th>w-kinney</th>
      <th>w-veselack</th>
      <th>w-mwhitt</th>
      <th>w-jarnold</th>
      <th>other,manager,specialist,director,executive</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>283</td>
      <td>c..koehler@enron.com</td>
      <td>128</td>
      <td>606</td>
      <td>734</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>1</th>
      <td>98</td>
      <td>scott.goodell@enron.com</td>
      <td>98</td>
      <td>607</td>
      <td>705</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>specialist</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
      <td>p..south@enron.com</td>
      <td>8</td>
      <td>351</td>
      <td>359</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>3</th>
      <td>204</td>
      <td>lavorato@enron.com</td>
      <td>388</td>
      <td>3</td>
      <td>391</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>executive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>318</td>
      <td>mike.grigsby@enron.com</td>
      <td>3702</td>
      <td>490</td>
      <td>4192</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>executive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>206</th>
      <td>114</td>
      <td>vkamins@enron.com</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>207</th>
      <td>270</td>
      <td>david.duran@enron.com</td>
      <td>7</td>
      <td>145</td>
      <td>152</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>208</th>
      <td>282</td>
      <td>sean.crandall@enron.com</td>
      <td>94</td>
      <td>138</td>
      <td>232</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director</td>
    </tr>
    <tr>
      <th>209</th>
      <td>243</td>
      <td>kevin.presto@enron.com</td>
      <td>248</td>
      <td>198</td>
      <td>446</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>executive</td>
    </tr>
    <tr>
      <th>210</th>
      <td>131</td>
      <td>dave.fuller@enron.com</td>
      <td>165</td>
      <td>129</td>
      <td>294</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>manager</td>
    </tr>
  </tbody>
</table>
<p>211 rows × 5120 columns</p>
</div>




```python
email_has_label_truth = ground_truth_email_nodes[ground_truth_email_nodes['id'].isin(list(email_nodes_targets['id']))][['id', 'title']].copy()
# Convert titles to integers so PSL can ground faster
email_has_label_truth['title'] = email_has_label_truth['title'].map(title_map)

# Add in an existence column
email_has_label_truth['exists'] = 1.0

full_set_email_has_label_truth = fill_observed_missing_possibilities(email_has_label_truth, ['id', 'title', 'exists'], list(title_map.values()) )
full_set_email_has_label_truth
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>exists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>303</td>
      <td>4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>27</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>269</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>231</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>155</th>
      <td>46</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>131</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>157</th>
      <td>201</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>158</th>
      <td>172</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>159</th>
      <td>241</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 3 columns</p>
</div>




```python
# Outputs to file
# full_set_email_has_label_truth.to_csv('EmailHasLabel_truth.csv', sep ='\t', index=False, header=False)
```

## Predicate: CoRef(E1, E2)


```python

```


```python
# Outputs to file
# coref_edges.to_csv('CoRef_obs.csv', sep ='\t', index=False, header=False)
```

## Predicate: EmailManages(E1, E2)


```python

```
