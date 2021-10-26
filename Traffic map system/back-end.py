import requests
import numpy as np
from datetime import timedelta, date, datetime
import json
from lxml import etree


url = "https://api.data.gov.hk/v1/historical-archive/get-file?url=http%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Fspeedmap.xml&time="

def getReqTimes(now):
    r = []
    r.append(now-timedelta(days=7))
    r.append(now-timedelta(days=1))
    r.append(now-timedelta(minutes=30))
    r.append(now-timedelta(minutes=20))
    r.append(now-timedelta(minutes=10))
    return r

def getReqData(time,length=10):
    data = {}
    for i in range(0,length):
        t = (time+timedelta(minutes=i)).strftime('%Y%m%d-%H%M')
        r = requests.get(url+t)
        tree = etree.XML(r.text.encode("utf-8"))
        for road in tree:
            value = {str(attr.tag.split('}')[-1]).lower(): attr.text for attr in road}
            link_id = value["link_id"]
            sp = value['traffic_speed']
            if(link_id not in data.keys()):
                data[link_id] = []
            data[link_id].append(int(sp))
    return {k:aggData(data[k],'mean') for k in data.keys()}

def aggData(arr,mode='mean'):
    if(mode=='mean'):
        return np.mean(arr)
    if(mode=='append_dict'):
        data = {}
        for aindex in range(len(arr)):
            for ak in arr[aindex].keys():
                if(ak not in data.keys()):
                    data[ak] = [-1,-1,-1,-1,-1]
                data[ak][aindex]=arr[aindex][ak]
        return data
def predict(data,model='HA'):
    return {k:round(aggData(list(filter(lambda x:x>0,data[k])),'mean'),2) for k in data.keys()}

if __name__ == "__main__":
    tm = datetime.now()+timedelta(hours=8)
    tm -= timedelta(minutes=tm.minute % 10,seconds=tm.second,microseconds=tm.microsecond)
    print('current time for prediction',tm)
    all_requested_time = getReqTimes(tm)
    print(all_requested_time)
    data = [getReqData(i) for i in all_requested_time]
    feature = aggData(data,'append_dict')
    result = predict(feature)
    fname = tm.strftime('%Y%m%d-%H%M')
    json.dump(result,open('/home/ubuntu/webapp/prediction/'+fname+'.pred','w'))