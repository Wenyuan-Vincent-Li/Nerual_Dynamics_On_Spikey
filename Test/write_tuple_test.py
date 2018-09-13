import csv

hashes = [('1C7E2110A55F42F525E196499FF88523', '4'), ('5A7E2110A55F42F525E196499FF88511', '89')]
hashdata = {}

for hashentry in hashes:
    hashdata[hashentry[0]] = hashentry

print hashdata
# with open('input.csv','r') as csvinput:
#     with open('output.csv', 'w') as csvoutput:
#         writer = csv.writer(csvoutput, lineterminator='\n')
#         reader = csv.reader(csvinput)


#         all = []
#         titlerow = next(reader)
#         titlerow.append('NUMBER')
#         all.append(titlerow)

#         for row in reader:
#             if row[0] in hashdata:
#                 row.append(hashdata[row[0]][1])
#                 all.append(row)
#             else:
#                 all.append(row)

#         writer.writerows(all)