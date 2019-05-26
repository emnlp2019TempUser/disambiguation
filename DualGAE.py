
from os.path import join
from authors import genMetrixData
from paper import genDocumenetMeterix
from GAE import train2
import codecs
from util import setting



wf = codecs.open(join(setting.resultPath, 'result.txt'), 'w', encoding='utf-8')

authornames = ['Hongbin Li', 'Hua Bai', 'Kexin Xu', 'Lin Huang', 'Lu Han', 'Min Zheng', 'Qiang Shi', 'Rong Yu', 'Tao Deng', 'Wei Quan', 'Xu Xu', 'Yanqing Wang', 'Yong Tian']

for authorname in authornames:
    wf.write('{} '.format(authorname))
    # First GCN
    genMetrixData.main(authorname)
    genDocumenetMeterix.main(authorname)
    name = authorname + '_document'

    # Second GCN
    Res, num_nodes, n_clusters = train2.gae_for_na(name, isend=True)
    prec, rec, f1 = Res[0], Res[1], Res[2]
    wf.write('prec: {}, rec: {}, f1: {}\n'.format(prec, rec, f1))

wf.close()

