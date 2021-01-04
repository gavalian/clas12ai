import sys

inpath = sys.argv[1]
outpath = sys.argv[2]

out = "{\n"
with open(inpath) as f:
    line = f.readline()
    line = line.replace('\'', '\"')
    line = line[1:]
    line = line[:-2]
    out += line + ",\n"
    out+= "\"events\":[\n"
    for line in f:
        line = line.replace('\'', '\"')
        out += (line[:-1]+ ',\n')

out = out[:-2]
out += "\n"
out += "]\n}"

with open(outpath, 'w') as f:
    f.write(out)
