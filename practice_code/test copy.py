from glob import glob

with open("inference.csv", "w") as f:
    f.write("")

txt_path = "/home/jhjeong/jiho_deep/dacon/MNIST_2/data/inference/"

for num in range(50000, 55000):
    file_name = txt_path + str(num) + ".png"
    txt_name = "/home/jhjeong/jiho_deep/dacon/MNIST_2/data/train_script/49379.txt"
    with open("inference.csv", "a") as ff:
        ff.write(file_name)
        ff.write(",")
        ff.write(txt_name)
        ff.write("\n")
'''  
    
'''

'''
with open("train.csv", "a") as f:
    f.write("")
'''


'''
def filenum_padding(filenum):
  
    if filenum < 10: 
        return '0000' + str(filenum)
    elif filenum < 100: 
        return '000' + str(filenum)
    elif filenum < 1000: 
        return '00' + str(filenum)
    elif filenum < 10000: 
        return '0' + str(filenum)
    else: 
        return str(filenum)

txt_path = "/home/jhjeong/jiho_deep/dacon/MNIST_2/data/train_script/"

with open("/home/jhjeong/jiho_deep/dacon/MNIST_2/data/train_script/dirty_mnist_2nd_answer.csv", "r") as f:
    lines = f.readlines()

    for line in lines:
        if line.strip().split(",")[0] == "index":
            pass
        
        else:
            file_name = filenum_padding(int(line.strip().split(",")[0]))           

            txt_path_final = txt_path + file_name + ".txt"
            print(file_name)
            
            
            with open(txt_path_final, "w") as f:
                f.write("")
            
            wow = line.strip().split(",")[1:]
            
            for i in wow:  
                with open(txt_path_final, "a") as f:
                    f.write(str(i))
                    f.write(",")
                
'''            
