import json
import pickle
import os
import time as T
import logging
import sys
import numpy as np
import scipy.sparse as sp
logging.basicConfig(filename="test.log", level=logging.DEBUG)
path='../'     

def getAdressaSubUser(filename, subusers):
    logging.debug("get all subscribed users in "+filename)
    for line in open(filename):
        data=line.rstrip()
        if "/pluss/" in line:
            d=json.loads(data)
            subusers.add(d['userId'])
    return subusers

def getAllUser(filename,userset):
    logging.debug("get all users in "+filename)
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        userset.add(d['userId'])
    return userset

def getAllUserAndUrl(filename,userset,urlset):
    logging.debug("get all urls and users"+filename)
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        userset.add(d['userId'])
        if d['url'] == 'http://adressa.no':
            continue
        elif 'ece' not in d['url'] and 'html' not in d['url']:
            continue
        urlset.add(d['url'])
    return userset,urlset

def getAdressaSubUserAndUrl(filename, subusers,urlset):
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        if "/pluss/" in line:
            subusers.add(d['userId'])
        # if url is a document page
        if d['url'] == 'http://adressa.no':
            continue
        elif 'ece' not in d['url'] and 'html' not in d['url']:
            continue
        urlset.add(d['url'])
    return subusers, urlset

def simplifyAdressa(filename, userEvent, userset, session):
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        if d['userId'] in userset:
            newd={}
            loc=[]
            activeTime=0
            if 'activeTime' in d.keys():
                activeTime=d['activeTime']
            if session.get(d['userId']) == None:
                session[d['userId']]=0
            else:
                if d['sessionStart']:
                    session[d['userId']]+=1
                    
            if d['url'] == 'http://adressa.no':
                continue
            elif 'ece' not in d['url'] and 'html' not in d['url']:
                continue
            newd['session']=session[d['userId']]
            newd['activeTime']=activeTime
            newd['url']=d['url']
            try:
                userEvent[d['userId']].append(newd)
            except:
                userEvent[d['userId']]=[newd]
    return userEvent, session

def filterbyactiveTime(time, userevt):
    logging.debug("Filter by active time(>"+str(time)+"s)")
    time=int(time)
    for user in userevt.keys():
        newevt=[]
        logging.debug(str(len(userevt[user])))
        for event in userevt[user]:
            if int(event['activeTime']) > time:
                newevt.append(event)
        userevt[user]=newevt
        logging.debug(str(len(userevt[user])))
    return userevt

# The activeTime for each article the user read throughout the 3 month is normalized by the largest activeTime of each user
def normalizeActiveTime(userevt):
    for usr in userevt:
        total=[]
        for evt in userevt[usr]:
            total.append(evt['activeTime'])
        for evt in userevt[usr]:
            evt['activeTime'] = float(evt['activeTime'])/max(total)
    return userevt
    


def main():
    subUserSet=set();urlSet=set()
    urlSet = pickle.load(open('urlSet.pkl', 'rb'))#, encoding='utf-8'
    print('urlSet loaded')
    subUserSet = pickle.load(open('subUserSet.pkl', 'rb'))
    print('subUserSet loaded')

    files=[]
    for month in [1,2,3]:
        for date in range(1,10):
            t='20170'+str(month)+'0'+str(date) 
            files.append(t)
        if month ==2:
            for date in range(10,29):
                t = '20170'+str(month)+str(date)
                files.append(t)      
        else:
            for date in range(10,32):
                t = '20170'+str(month)+str(date)
                files.append(t)

    idx_to_url=list(urlSet);
    url_to_idx = dict([(url, i) for i, url in enumerate(idx_to_url)])
    idx_to_user=list(subUserSet);
    user_to_idx = dict([(usr, i) for i, usr in enumerate(idx_to_user)])
    #usr_url_matrix = np.zeros((len(idx_to_user),len(idx_to_url)))
    #print(usr_url_matrix.shape)

    for filename in files:
        os.system('tar -zxvf '+path+'three_month.tar.gz '+filename)
        print('tar -zxvf '+path+'three_month.tar.gz '+filename)
        f=filename
        for line in open(f):
            data=line.rstrip()
            d=json.loads(data)
            if 'activeTime' in d.keys() and d['activeTime']>0:
                if d['userId'] in subUserSet：
                    if d['url'] == 'http://adressa.no':
                            continue
                    elif 'ece' not in d['url'] and 'html' not in d['url']:
                            continue
                    user_idx = user_to_idx[d['userId']]
                    url_idx=url_to_idx[d['url']]
                    #usr_url_matrix[user_idx,url_idx]=d['activeTime']
                    with open("users.txt", "a") as user_file:
                        user_file.write(str(user_idx)+" ")
                    with open("items.txt", "a") as item_file:
                        user_file.write(str(url_idx)+" ")
                    with open("ratings.txt", "a") as rating_file:
                        user_file.write(str(d['activeTime'])+" ")                                                
        os.system('rm '+filename)
        print('remove '+filename)
    print("finish building matrix")
    
    # usr_url_matrix_csr = sp.csr_matrix(usr_url_matrix)
    # np.save("user_url_mat.npy",usr_url_matrix_csr)
    # url_usr_matrix = usr_url_matrix.transpose()
    # url_usr_matrix_csr = sp.csr_matrix(url_usr_matrix)
    # np.save("url_usr_mat.npy",url_usr_matrix_csr)
        
if __name__ == "__main__":
    main()    
