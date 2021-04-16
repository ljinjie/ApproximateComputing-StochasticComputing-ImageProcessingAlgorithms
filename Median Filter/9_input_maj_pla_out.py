def find_output(n):
    n = str(bin(n))
    print(n)
    one_count = 0
    for i in n:
        if i == "1":
            one_count += 1
    if one_count > 4:
        return 1
    else:
        return 0


if __name__ == '__main__':
    file1 = open("9_input_maj.txt","w")
    for i in range(512):
        file1.write(format(i, '09b') + ' ' + str(find_output(i)) + '\n')
    file1.close()
