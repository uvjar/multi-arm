import json
import pickle
import os
import time as T
import logging
import sys
import numpy as np
import argparse
import os.path
import scipy.sparse as sp
logging.basicConfig(filename="test.log", level=logging.DEBUG)
   

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



def main():
    parser = argparse.ArgumentParser(description='get user, url and rating file')
    parser.add_argument("--input_file", help="input_file. three_month.tar.gz or one_week.tar.gz", default='../../dataset/one_week.tar.gz')
    parser.add_argument("--output_path", help='output directory', default='build')
    args = parser.parse_args()

    output_path=args.output_path;input_path=args.input_file; 
    if os.path.isfile(args.input_file)==False:
        print("tar file does not exist")
        exit(1)


    files=[]
    if input_path.endswith("three_month.tar.gz"):
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
    elif input_path.endswith("one_week.tar.gz"):
        for date in range(1,8):
            t='2017010'+str(date) 
            files.append(t)
    print(files)

    subUserSet=set();
    for filename in files:
        os.system('tar -zxvf '+input_path+' '+filename)
        print('tar -zxvf '+input_path+' '+filename)
        subUserSet = getAdressaSubUser(filename, subUserSet);
        os.system('rm '+filename)
        print('remove '+filename)
    print("finish getting user set")
               
    print("total amount of subscribing users: "+str(len(subUserSet)))
    with open(output_path+'/subUserSet.pkl', 'wb') as f:
        pickle.dump(subUserSet, f)
        

if __name__ == "__main__":
    main()            





