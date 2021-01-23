from Q7_IncreSim import veoScore, deltaConSim, increSI


if __name__ == "__main__":
    filepath_list = []
    for i in range(29):
        filepath_list.append(f'datasets/temporal-fb/fb-{i}.txt')
    
    name_list = ['from', 'to', 'weight', 'timestamp']

    print(increSI.GraphSimilarity(filepath_list, name_list).run())
    print(veoScore.GraphSimilarity(filepath_list, name_list).run())
    print(deltaConSim.GraphSimilarity(filepath_list, name_list).run())
