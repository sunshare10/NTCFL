# # import 
# fr = open("markdict.txt",'r+')
# dic = eval(fr.read()) #读取的str转换为字典
# print(dic)
# fr.close()



with open('markdict.txt','r+') as fr:
     classdict=eval(fr.read())
classes_names,indexs=[],[]
classes_names=list(classdict.keys())
indexs=list(classdict.values())
print(classes_names)
print(indexs)