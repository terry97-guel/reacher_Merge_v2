from envs.reacherEnv import get_quadrant
from matplotlib import pyplot as plt
from utils.dataloader import Buffer, Sampler
from utils.tools import make_circle, cast_dict_numpy
import numpy as np
from pathlib import Path

def get_measure(batch, mode:int,  PLOT:bool, plot_name:Path):
    """
    input[0]: batch
    
    return[0]: measure_dict:Dict{
        key[0]: Accuracy            - total accuracy
        key[1]: Accuracy_Q1         - Quadrant1 accuracy
        key[2]: Accuracy_Q2         - Quadrant2 accuracy
        key[3]: Accuracy_Q3         - Quadrant3 accuracy
        key[4]: Accuracy_Q4         - Quadrant4 accuracy
        key[5]: Var                 - Variance of last position
        key[6]: Coverage           - Coverage of last position
    }
    """
    length = len(batch['anchor'])
    success_per_quad  = {0:[],1:[],2:[],3:[]}
    correct_points = []
    wrong_points = []
    
    for target_quadrant,last_position in zip(batch["target_quadrant"], batch["last_position"]):
        last_quadrant = get_quadrant(last_position)

        success_per_quad[target_quadrant].append(last_quadrant == target_quadrant)
        
        if last_quadrant == target_quadrant:
            correct_points.append(last_position)
        else:
            wrong_points.append(last_position)
    
    if len(correct_points) == 0:
        return {"Accuracy":.0, "Var":.0, "Coverage":.0, "Accuracy_Q1":.0, "Accuracy_Q2":.0, "Accuracy_Q3":.0, "Accuracy_Q4":.0}
    else: correct_points = np.stack(correct_points)
    
    if len(wrong_points) == 0: wrong_points = None
    else: wrong_points = np.stack(wrong_points)
    
    measure_dict = {}
    
    # Measure accuracy
    success = 0
    for idx,success_flag in success_per_quad.items():
        if len(success_flag) == 0:
            accuracy_quad = .0
        else:
            success_quad = np.sum(success_flag)
            accuracy_quad = success_quad/len(success_flag)
            success = success + success_quad
            
        measure_dict.update({f"Accuracy_Q{idx+1}": accuracy_quad})
    
    accuracy = success/length
    measure_dict.update({"Accuracy": accuracy})
    
    # Measure Variance and Coverage
    var, coverage, cluster_idxs, hulls = measure_var_coverage(correct_points, mode)
    
    if PLOT:
        plot_QD_figure(correct_points, wrong_points, cluster_idxs, mode, accuracy, coverage, hulls, PLOT, plot_name)
    measure_dict.update({"Var": var, "Coverage": coverage})

    return cast_dict_numpy(measure_dict)


def plot_QD_figure(correct_points, wrong_points, cluster_idxs, mode, accuracy, coverage, hulls:tuple, PLOT:bool, plot_name:Path):
    plt.figure(figsize=(10,10))
    
    for k_ in range(mode):
        cluster_points = correct_points[cluster_idxs == k_]
        cluster_points = np.unique(cluster_points,axis=0)
        if PLOT:
            # PLOT Each Cluster
            plt.scatter(cluster_points[:,0],cluster_points[:,1], color = 'k')

        # Plot Convex Hull
        hull = hulls[k_]
        if hull is not None:
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')
    
    if wrong_points is not None:
        plt.plot(wrong_points[:,0],wrong_points[:,1],'x',color='r',label="Wrong")

    name = "Convex Area"
    plt.title(name, fontsize=20)
    plt.legend()
    plt.xlim(-0.22, 0.22)
    plt.ylim(-0.22, 0.22)

    a,b = make_circle(radius=0.09)
    plt.plot(a,b, color='k')
    a,b = make_circle(radius=0.21)
    plt.plot(a,b, color='k')

    plt.xlabel("X-axis", fontsize=14)
    plt.ylabel("Y-axis", fontsize=14)

    font1 = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 16}

    plt.text(0.10,-0.212,"Coverage:{:.2f}%".format(coverage*100), fontdict = font1)
    plt.text(0.101,-0.196,"Accuracy:{:.2f}%".format(accuracy*100), fontdict = font1)
    
    Path(plot_name.parent).mkdir(exist_ok=True)
    if plot_name.suffix != ".png":
        plot_name = plot_name.with_suffix(".png")
    
    plt.savefig(plot_name.__str__())

def measure_var_coverage(correct_points, mode):
    from matplotlib import pyplot as plt
    from scipy.spatial import ConvexHull
    from scipy.cluster.vq import kmeans
    from scipy.spatial.distance import cdist
    
    hulls = []
    
    coverage = .0
    var      = .0
    
    codebook, distortion = kmeans(correct_points, mode)
    dist_matrix = cdist(correct_points, codebook)
    cluster_idxs = np.argmin(dist_matrix, axis=1)

    for k_ in range(mode):
        cluster_points = correct_points[cluster_idxs == k_]
        cluster_points = np.unique(cluster_points,axis=0)
        
        if len(cluster_points) <3: 
            hulls.append(None)
            continue
        else:
            hull = ConvexHull(cluster_points)
            hulls.append(hull)

        
        for simplex in hull.simplices[1:]:
            triangle = []
            triangle.append(np.concatenate([cluster_points[hull.simplices[0]][0], np.array([1])]))
            triangle.append(np.concatenate([cluster_points[simplex][0], np.array([1])]))
            triangle.append(np.concatenate([cluster_points[simplex][1], np.array([1])]))
            triangle = np.stack(triangle)

            coverage = coverage + np.abs(np.linalg.det(triangle)/2)
        
        var = var + np.std(cluster_points)

    WS_AREA = (0.21**2-0.09**2) * np.pi
    coverage = coverage / WS_AREA 


    return var, coverage, cluster_idxs, tuple(hulls)