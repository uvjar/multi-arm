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


    for filename in files:
        os.system('tar -zxvf '+path+'three_month.tar.gz '+filename)
        print('tar -zxvf '+path+'three_month.tar.gz '+filename)
        f=filename
        subUserSet,urlSet=getAdressaSubUserAndUrl(f,subUserSet,urlSet) 
        os.system('rm '+filename)
        print('remove '+filename)
    print("finish reading user set and item set")
               
    logging.debug("total amount of subscribing users: "+str(len(subUserSet)))
    print("total amount of subscribing users: "+str(len(subUserSet)))
    with open('subUserSet.pkl', 'wb') as f:
        pickle.dump(subUserSet, f)
    with open('urlSet.pkl', 'wb') as f:
        pickle.dump(urlSet, f)
        

if __name__ == "__main__":
    main()            





