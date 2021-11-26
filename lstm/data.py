#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb

#%%
#Get data(currency_statistics)
data= pd.read_csv("data/currency_statistics/market-price")
start=end=Sum=0
X= np.array([])
tstmp=np.array([])
tempDate= data['timestamp'][0][0:11]
tstmp= data['timestamp'][0][0:11]
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        tstmp= np.append(tstmp,data['timestamp'][ts][0:11])
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_price=X

X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
price = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/wallet_activity/my-wallet-n-users")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
print(X)
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_TW=X

X = np.diff(np.log(np.array(X))).reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
wallet_activity = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/currency_statistics/trade-volume")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
volumeX=X

X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
trade_volume = X_scaled.reshape(1,-1)[0]

# data= pd.read_csv("data/currency_statistics/market-cap")
# start=end=Sum=0
# tempDate = data['timestamp'][0][0:11]
# X= np.array([])
# for ts in range(data.shape[0]):
#     if tempDate == data['timestamp'][ts][0:11]:
#         end= ts
#         Sum+= data['value'][ts]
#     else:
#         X = np.append(X, Sum/(end - start + 1))
#         start= end= ts
#         tempDate= data['timestamp'][ts][0:11]
#         Sum= data['value'][ts]
# markcap=X
# X = np.log(np.array(X)).reshape(-1,1) 
# scaler = MinMaxScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
# market_cap = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/currency_statistics/total-bitcoins")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_TB = X
X = np.diff(np.log(np.array(X))).reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
total_bitcoins = X_scaled.reshape(1,-1)[0]

#plot scaled(currency_statistics)
plt.figure(figsize=(20,10))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price', marker='.')
plt.plot(range(trade_volume.shape[0]), trade_volume  , color='#FF0000', label='Trade volume')
# plt.plot(range(market_cap.shape[0]), market_cap     , color='#008000', label='Market cap')
plt.plot(range(total_bitcoins.shape[0]), total_bitcoins, color='#FFFF00', label='Total bitcoins')
plt.plot(range(wallet_activity.shape[0]), wallet_activity, color='#66FF44', label='Wallet Activity (blockchain.com')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Currency Statistics")
plt.legend()
plt.grid()
plt.show()

#Correlation (currency_statistics)              #pd.DataFrame(market_cap)
X=pd.concat([pd.DataFrame(trade_volume),   pd.DataFrame(total_bitcoins),  pd.DataFrame(wallet_activity)],axis=1)
Y= pd.DataFrame(price)
df=pd.DataFrame(data=X)
df.columns =['trade_volume', 'total_bitcoins','Wallet Activity']
df['Price']=Y
plt.figure(figsize=(15,8))
ttl=sb.heatmap(df.corr(),annot=True)
ttl.set_title("Currency Statistics", fontsize=20, fontweight="bold")

#%%
#Get data(block_details)
data= pd.read_csv("data/block_details/avg-block-size")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
avg_block_size = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/avg-confirmation-time")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
avg_confirmation_time = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/blocks-size")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_BS=X
X = np.diff(np.log(np.array(X))).reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
blocks_size = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/median-confirmation-time")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_MCT=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
median_confirmation_time = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/n-payments-per-block")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_payments_per_block = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/n-transactions-per-block")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_transactions_per_block = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/block_details/n-transactions-total")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_TNT=X
X = np.diff(np.log(np.array(X))).reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_transactions_total = X_scaled.reshape(1,-1)[0]

#plot scaled(block_details)
#1
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(avg_block_size.shape[0]), avg_block_size  , color='#000000', label='AVG Block Size')
plt.plot(range(avg_confirmation_time.shape[0]), avg_confirmation_time  , color='#FF0000', label='AVG Confirmation Time')
plt.plot(range(blocks_size.shape[0]), blocks_size  , color='#FFFF00', label='Blocks Size')
plt.plot(range(median_confirmation_time.shape[0]), median_confirmation_time  , color='#00FF00', label='Median Confirmation Time')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Block Details(1)")
plt.legend()
plt.grid()
plt.show()
#2
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(n_payments_per_block.shape[0]), n_payments_per_block  , color='#00FFFF', label='n_payments_per_block')
plt.plot(range(n_transactions_per_block.shape[0]), n_transactions_per_block  , color='#FF00FF', label='n_transactions_per_block')
plt.plot(range(n_transactions_total.shape[0]), n_transactions_total  , color='#008000', label='n_transactions_total')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Block Details(2)")
plt.legend()
plt.grid()
plt.show()

#Correlation (block_details)
X=pd.concat([pd.DataFrame(avg_block_size), pd.DataFrame(avg_confirmation_time), pd.DataFrame(blocks_size), pd.DataFrame(median_confirmation_time), pd.DataFrame(n_payments_per_block), pd.DataFrame(n_transactions_per_block), pd.DataFrame(n_transactions_total)],axis=1)
Y= pd.DataFrame(price)
df=pd.DataFrame(data=X)
df.columns =['avg_block_size','avg_confirmation_time','blocks_size','median_confirmation_time','n_payments_per_block','n_transactions_per_block','n_transactions_total']
df['Price']=Y
plt.figure(figsize=(20,15))
ttl=sb.heatmap(df.corr(),annot=True)
ttl.set_title("Block Details", fontsize=20, fontweight="bold")

#%%
#Get data(market_signals)
data= pd.read_csv("data/market_signals/mvrv")
X = np.array(data[14:1478:4]['value']).reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
mvrv = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/market_signals/nvt")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_MVRV=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
nvt = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/market_signals/nvts")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
nvts = X_scaled.reshape(1,-1)[0]

#plot scaled(market_signals)
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(mvrv.shape[0]), mvrv  , color='#00FF00', label='Market Value to Realised Value')
plt.plot(range(nvt.shape[0]), nvt  , color='#FF0000', label='Network Value to Transactions')
plt.plot(range(nvts.shape[0]), nvts  , color='#FFFF00', label='Network Value to Transactions Signal')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Market Signals")
plt.legend()
plt.grid()
plt.show()

#Correlation (market_signals)
X=pd.concat([pd.DataFrame(mvrv), pd.DataFrame(nvt), pd.DataFrame(nvts)],axis=1)
Y= pd.DataFrame(price)
df=pd.DataFrame(data=X)
df.columns =['Market Value to Realised Value', 'Network Value to Transactions', 'Network Value to Transactions Signal']
df['Price']=Y
plt.figure(figsize=(15,8))
ttl=sb.heatmap(df.corr(),annot=True)
ttl.set_title("Market Signals", fontsize=20, fontweight="bold")


#%%
#Get data(mining_information)
data= pd.read_csv("data/mining_information/cost-per-transaction")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
cost_per_transaction = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/cost-per-transaction-percent")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
cost_per_transaction_percent = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/difficulty")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_dif=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
difficulty = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/hash-rate")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_HR=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
hash_rate = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/fees-usd-per-transaction")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fees_usd_per_transaction = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/miners-revenue")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
miners_revenue = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/transaction-fees")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
transaction_fees = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/mining_information/transaction-fees-usd")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
transaction_fees_usd = X_scaled.reshape(1,-1)[0]

#plot scaled(mining_information)
#1
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(cost_per_transaction.shape[0]), cost_per_transaction  , color='#000000', label='Cost Per Transaction')
plt.plot(range(cost_per_transaction_percent.shape[0]), cost_per_transaction_percent  , color='#FFFF00', label='cost/transaction %')
plt.plot(range(difficulty.shape[0]), difficulty  , color='#FF0000', label='Difficulty')
plt.plot(range(hash_rate.shape[0]), hash_rate  , color='#00FF00', label='Hash_rate')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Mining Information(1)")
plt.legend()
plt.grid()
plt.show()
#2
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(fees_usd_per_transaction.shape[0]), fees_usd_per_transaction  , color='#00FFFF', label='fees_usd/transaction')
plt.plot(range(miners_revenue.shape[0]), miners_revenue  , color='#FF00FF', label='miners revenue')
plt.plot(range(transaction_fees.shape[0]), transaction_fees  , color='#008000', label='transaction fees')
plt.plot(range(transaction_fees_usd.shape[0]), transaction_fees_usd  , color='#00FF00', label='transaction_fees_usd')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Mining Information(2)")
plt.legend()
plt.grid()
plt.show()

#Correlation (market_signals)
X=pd.concat([pd.DataFrame(cost_per_transaction), pd.DataFrame(cost_per_transaction_percent), pd.DataFrame(difficulty), pd.DataFrame(hash_rate), pd.DataFrame(fees_usd_per_transaction), pd.DataFrame(miners_revenue), pd.DataFrame(transaction_fees), pd.DataFrame(transaction_fees_usd)],axis=1)
Y= pd.DataFrame(price)
df=pd.DataFrame(data=X)
df.columns =['transaction_fees_usd','transaction fees','miners revenue','fees_usd/transaction','Hash_rate','Difficulty','cost/transaction %','Cost Per Transaction']
df['Price']=Y
plt.figure(figsize=(20,15))
ttl=sb.heatmap(df.corr(),annot=True)
ttl.set_title("Mining Information", fontsize=20, fontweight="bold")

#%%
#Get data(network_activity)
data= pd.read_csv("data/network_activity/estimated-transaction-volume")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
estimated_transaction_volume = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/estimated-transaction-volume-usd")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
estimated_transaction_volume_usd = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/mempool-count")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
mempool_count = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/mempool-growth")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
mempool_growth = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/mempool-size")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_memsize=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
mempool_size = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/n-payments")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_payments = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/n-transactions")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_transactions = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/n-transactions-excluding-popular")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_transactions_excluding_popular = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/n-unique-addresses")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
n_unique_addresses = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/output-volume")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
output_volume = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/transactions-per-second")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
transactions_per_second = X_scaled.reshape(1,-1)[0]

data= pd.read_csv("data/network_activity/utxo-count")
start=end=Sum=0
tempDate = data['timestamp'][0][0:11]
X= np.array([])
for ts in range(data.shape[0]):
    if tempDate == data['timestamp'][ts][0:11]:
        end= ts
        Sum+= data['value'][ts]
    else:
        X = np.append(X, Sum/(end - start + 1))
        start= end= ts
        tempDate= data['timestamp'][ts][0:11]
        Sum= data['value'][ts]
csv_utxo=X
X = np.log(np.array(X)).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
utxo_count = X_scaled.reshape(1,-1)[0]

#plot scaled(network_activity)
#1
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(mempool_size.shape[0]), mempool_size  , color='#00FFFF', label='mempool_size')
plt.plot(range(mempool_count.shape[0]), mempool_count  , color='#FF0000', label='mempool_count')
plt.plot(range(mempool_growth.shape[0]), mempool_growth  , color='#00FF00', label='mempool_growth')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Network Activity(1)")
plt.legend()
plt.grid()
plt.show()
#2
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(estimated_transaction_volume.shape[0]), estimated_transaction_volume  , color='#000000', label='estimated_transaction_volue')
plt.plot(range(n_payments.shape[0]), n_payments  , color='#FF00FF', label='n_payments')
plt.plot(range(n_transactions.shape[0]), n_transactions  , color='#008000', label='n_transactions')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Network Activity(2)")
plt.legend()
plt.grid()
plt.show()
#3
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(n_unique_addresses.shape[0]), n_unique_addresses  , color='#00FFFF', label='n_unique_addresses')
plt.plot(range(output_volume.shape[0]), output_volume  , color='#FF00FF', label='output_volume')
plt.plot(range(utxo_count.shape[0]), utxo_count  , color='#00FF00', label='Unspent Transaction Outputs(UTXO)')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Network Activity(3)")
plt.legend()
plt.grid()
plt.show()
#4
plt.figure(figsize=(30,15))
plt.plot(range(price.shape[0]), price  , color='#000080', label='Price',marker='.' )
plt.plot(range(estimated_transaction_volume_usd.shape[0]), estimated_transaction_volume_usd  , color='#FFFF00', label='estimated_transaction_volume_usd')
plt.plot(range(n_transactions_excluding_popular.shape[0]), n_transactions_excluding_popular  , color='#00FF00', label='n_transactions_excluding_popular')
plt.plot(range(transactions_per_second.shape[0]), transactions_per_second  , color='#ff0000', label='transactions_per_second')
plt.xlabel('Steps (2020/09/20 - 2021/09/21)', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title("Network Activity(4)")
plt.legend()
plt.grid()
plt.show()

#Correlation (network_activity)
X=pd.concat([pd.DataFrame(mempool_size), pd.DataFrame(mempool_count),pd.DataFrame(mempool_growth), pd.DataFrame(estimated_transaction_volume), pd.DataFrame(n_payments), pd.DataFrame(n_transactions), pd.DataFrame(n_unique_addresses), pd.DataFrame(output_volume), pd.DataFrame(utxo_count), pd.DataFrame(estimated_transaction_volume_usd), pd.DataFrame(n_transactions_excluding_popular), pd.DataFrame(transactions_per_second)],axis=1)
Y= pd.DataFrame(price)
df=pd.DataFrame(data=X)
df.columns =['mempool_size','mempool_count','mempool_growth','estimated_transaction_volue','n_payments','n_transactions','n_unique_addresses','output_volume','Unspent Transaction Outputs(UTXO)','estimated_transaction_volume_usd','n_transactions_excluding_popular','transactions_per_second']
df['Price']=Y
plt.figure(figsize=(20,15))
ttl=sb.heatmap(df.corr(),annot=True)
ttl.set_title("Network Activity", fontsize=20, fontweight="bold")

#%%
#save Best Features to CSV file
# from numpy import savetxt

# X=np.array([])
# X=pd.concat([ pd.DataFrame(tstmp),pd.DataFrame(csv_price),pd.DataFrame(csv_dif), pd.DataFrame(csv_utxo), pd.DataFrame(csv_TW), pd.DataFrame(csv_TNT), pd.DataFrame(csv_BS), pd.DataFrame(csv_TB), pd.DataFrame(csv_HR), pd.DataFrame(csv_MVRV), pd.DataFrame(csv_MCT), pd.DataFrame(csv_memsize),pd.DataFrame(volumeX)],axis=1)
# df=pd.DataFrame(data=X)
# df.columns =['timeStamp','price','difficulty', 'utxo', 'wallet', 'total_transaction', 'block_size', 'total_btc', 'hash_rate', 'MVRV', 'median_confirmation_time', 'mempool_size','volume' ]

# df.to_csv('data/top_features.csv' ,sep=',' ,index=True)

# # savetxt('data/top_features.csv', df, delimiter=',')