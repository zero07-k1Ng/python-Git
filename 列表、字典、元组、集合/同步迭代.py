counts=[1,3,5,7]
fruits={'apple','banana','pear','dick'}
for c,f in zip(counts,fruits):
    match c,f:
        case 1,'apple':
            print('1个苹果')
        case 3,'banana':
            print('3个香蕉')
        case 5,'pear':
            print('5个梨')
        case 7,'dick':
            print('7个格调')