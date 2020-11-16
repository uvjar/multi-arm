import pickle
import os
import argparse
import json
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='get user, url and rating file')
    parser.add_argument("--input_file", help="input_file. three_month.tar.gz or one_week.tar.gz", default='../three_month.tar.gz')
    parser.add_argument("--output_path", help='output directory', default='data')
    parser.add_argument("--userSet_path", help='userSet.pkl directory', default='subUserSet.pkl')
    args = parser.parse_args()

    output_path=args.output_path;input_path=args.input_file; userSet_path=args.userSet_path;
    if os.path.isfile(args.input_file)==False:
        print("tar file does not exist")
        exit(1)
    if os.path.isfile(args.userSet_path)==False:
        print("pkl file does not exist")
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

    subUserSet = pickle.load(open(userSet_path, 'rb'))
    print('subUserSet loaded, '+str(len(subUserSet))+' subscribed users in total')
    idx_to_user=list(subUserSet);
    user_to_idx = dict([(usr, i) for i, usr in enumerate(idx_to_user)])

    
    urlSet=[];
    for filename in files:
        if os.system('tar -zxvf '+input_path+' '+filename)!=0:
            print("Fail: "+'tar -zxvf '+input_path+' '+filename);
            exit(1);
        print('tar -zxvf '+input_path+' '+filename)
        for line in open(filename):
            data=line.rstrip()
            d=json.loads(data)
            if d['url'] == 'http://adressa.no':
                continue
            elif 'ece' not in d['url'] and 'html' not in d['url']:
                continue
            if 'activeTime' in d.keys():
                if d['userId'] in subUserSet:
                    try:
                        url_idx=urlSet.index( d['url']   )
                    except:
                        urlSet.append(d['url']);
                        url_idx=urlSet.index(d['url'])

                    user_idx = user_to_idx[d['userId']];
                    with open(output_path+"/users.txt", "a") as user_file:
                        user_file.write(str(user_idx)+" ")
                    with open(output_path+"/items.txt", "a") as item_file:
                        item_file.write(str(url_idx)+" ")
                    with open(output_path+"/ratings.txt", "a") as rating_file:
                        rating_file.write(str(d['activeTime'])+" ")    

        os.system('rm '+filename)
        print('remove '+filename)
    print("finish reading user set and item set")
        
        
if __name__ == "__main__":
    main()    
