
def dev_phase(yoy, ebitda):
    if yoy == '-' or ebitda == '-':
        return '-'
    else:
        yoy = float(yoy)
        ebitda = float(ebitda)
        if yoy < -10:
            return 1
        elif -10 <= yoy < -5:
            if ebitda < -5:
                return 1
            else:
                return 2
        elif -5 <= yoy < 0:
            if ebitda < -10:
                return 1
            else:
                return 2
        elif 0 <= yoy < 10:
            if ebitda < -20:
                return 1
            else:
                return 2
        elif 10 <= yoy < 20:
            if ebitda < -20:
                return 1
            elif ebitda < 10:
                return 2
            else:
                return 3
        elif 20 <= yoy < 30:
            if ebitda < -20:
                return 1
            elif ebitda < -5:
                return 2
            else:
                return 3
        elif yoy >= 30:
            if ebitda < -20:
                return 1
            elif ebitda < -5:
                return 2
            else:
                return 3


def share_bonus(payout, repurchase):
    if payout == '-' or repurchase == '-':
        return '-'
    else:
        payout = int(float(payout))
        repurchase = int(float(repurchase))
        if payout == 0 and repurchase == 0:
            return 0
        if payout > 0 or repurchase > 0:
            return 1



if __name__ == '__main__':
    # yoy_ebitda_input = open('./yoy_ebitda.csv')
    # lines = yoy_ebitda_input.readlines()
    # for line in lines[1:]:
    #     items = line.split(',')
    #     col_num = len(items)
    #     for i in xrange(20):
    #         ii = i + 20
    #         print str(dev_phase(items[i].strip(), items[ii].strip())) + '\t',
    #     print

    payout_repurchase_input = open('./payoutRatio_repurchase.csv')
    lines = payout_repurchase_input.readlines()
    for line in lines[1:]:
        items = line.split(',')
        for i in xrange(20):
            ii = i + 20
            print str(share_bonus(items[i].strip(), items[ii].strip())) + '\t',
        print