import csv
paths = [('pedal', 'p1'), ('non-pedal', 'p2')]
for state, path in paths:
    print(state)
    print(path)

'''
path = '/ssd2/maestro/maestro-v1.0.0/yj_dataset/converted_wavefile/'
files = ['test_onset_p', 'test_onset_np', 'train_onset_p', 'train_onset_np', 'test_segment_p', 'test_segment_np', 'train_segment_p', 'train_segment_np']
for file_name in files:
    print(file_name)
    f = open(path + file_name + '.csv', 'r')
    wf = open(path + file_name + '_.csv', 'w')
    reader = csv.reader(f)
    writer = csv.writer(wf)
    for line in reader:
        writer.writerow([line[0].split('.wav')[0], line[1], line[2]])

    f.close()
    wf.close()



a = dict()
print(a)

a['p'] = dict()
a['np'] = dict()
print(a)

p_dict = a['p']
np_dict = a['np']

p_dict['n1'] = (1,2)
np_dict['n1'] = (1,2)

print(a)
'''
'''
list = []
print(list)
list = zip(['n1', '0', '1'], ['n2', '5', '6'])
print(list)

file_names = []
starts = []
ends = []

f = open('tmp.csv', 'w')
writer = csv.writer(f)
for i in range(0,10):
    file_name = "n" + str(i)
    file_names.append(file_name)
    start = str(2*i)
    starts.append(start)
    end = str(2*i + 1)
    ends.append(end)
    writer.writerow([file_name, start, end])
    i = i + 1
print(zip(*[file_names, starts, ends]))


f.close()

f = open('tmp.csv', 'r')
rdr = csv.reader(f)
for line in rdr:
    print(line)
    print(line[0])
f.close()
'''