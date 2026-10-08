import undercode

ai = undercode.AI([1,3,10,10,3,1])
md = input("Mode> ")
if md == '0':
    while 1:
        ask_data = input("> ").split(' ')
        if ask_data == ['']:
            continue
        for i in undercode.rl(ask_data):
            ask_data[i] = float(ask_data[i])
        print(*ai.ask(ask_data))
elif md == '1':
    ai.teach_cycle()