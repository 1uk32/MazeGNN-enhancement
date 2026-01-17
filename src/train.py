import time
import copy
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from model import *

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(dataset, args):
    test_loader = loader = DataLoader(dataset, batch_size=32, shuffle=True) #somehow it does not batch correctly

    # build model
    model = GNNStack(args.node_feat_dim, args.hidden_dim, args.num_clusters, args)
    model.loss = args.lossft
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_loss = []
    best_loss = float('inf') # Initialize best_loss properly
    best_model =  copy.deepcopy(model)
    t0=time.perf_counter()
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred_softmax, pred_embeddings = model(batch) # Unpack outputs
            # Use pred_softmax for cluster assignment and pred_embeddings for coordinates in kmeansloss
            loss = model.loss(pred_softmax[batch.train_mask], pred_embeddings[batch.train_mask])
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            if args.visualize:
                # For training visualization, use the pred_softmax and pred_embeddings from the last batch of the current epoch
                visualize(pred_softmax, pred_embeddings, batch, batch.train_mask)
            if args.test:
                # The test function now returns both the total loss, the last pred_softmax, and last pred_embeddings for visualization
                current_test_loss, pred_softmax_test_batch, pred_embeddings_test_batch = test(test_loader, model)
                test_loss.append(current_test_loss)
                t1=time.perf_counter()
                print('ep:',epoch, ', train loss', loss.item(), ', test loss:', current_test_loss, t1-t0, 's.')
                if current_test_loss < best_loss:
                    best_loss = current_test_loss
                    best_model = copy.deepcopy(model)
                if args.visualize:
                    # For test visualization, use the pred_softmax and pred_embeddings returned from the test function
                    visualize(pred_softmax_test_batch, pred_embeddings_test_batch, batch, batch.test_mask)
        else:
            if args.test:
                test_loss.append(test_loss[-1])

    if not args.test:
        best_model = copy.deepcopy(model) #just return the final trained model if there is no test

    return test_loss, losses, best_model, best_loss, test_loader

def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):
    test_model.eval()
    total_loss=0
    last_pred_softmax = None # To store the pred_softmax for the last processed batch for visualization
    last_pred_embeddings = None # To store the pred_embeddings for the last processed batch for visualization
    for batch in loader:
        with torch.no_grad():
            pred_softmax, pred_embeddings = test_model(batch) # Unpack outputs
        mask = batch.val_mask if is_validation else batch.test_mask
        # Use pred_softmax for cluster assignment and pred_embeddings for coordinates in kmeansloss
        loss = test_model.loss(pred_softmax[mask], pred_embeddings[mask])
        total_loss += loss.item() * batch.num_graphs
        last_pred_softmax = pred_softmax # Store the prediction for visualization
        last_pred_embeddings = pred_embeddings # Store the embeddings for visualization
    total_loss /= len(loader.dataset)
    return total_loss, last_pred_softmax, last_pred_embeddings # Return both total_loss, the last pred_softmax, and last pred_embeddings


def visualize(pred_softmax, pred_embeddings, data, mask, overlay=False): # Updated signature
    """
    pred_softmax: softmax output (cluster assignment probabilities)
    pred_embeddings: raw embeddings from the final linear layer (used for centroid calculation)
    data: data object (contains original node coordinates for plotting)
    mask: train or test mask
    """
    if type(mask)!=list:
        mask = list(mask)
    original_x = data.x*data.scaler+data.shift # Use original_x for plotting nodes on the 2D maze
    ## soft centroid
    cluster_size = torch.sum(pred_softmax[mask],axis=0).reshape(pred_softmax[mask].shape[1],1) # Use pred_softmax for cluster assignment
    # Centroids are now calculated using embeddings, which can be high-dimensional
    centroids = torch.div(torch.matmul(pred_softmax[mask].t(), pred_embeddings[mask]), cluster_size).detach().numpy() # Use pred_embeddings here
    colors=['r','orange','yellow','g','b','magenta','purple','turquoise']
    num_clusters = pred_softmax.shape[1] # Use pred_softmax here
    if num_clusters>=4:
        rows = num_clusters//4+1 if num_clusters%4!=0 else num_clusters//4
        cols = 4
        aspectratio = rows/cols
    else:
        rows=1
        cols=num_clusters

    if overlay:
        fig2 =plt.figure(figsize=(6, 6))

        for k in range(pred_softmax.shape[1]): # Use pred_softmax here
            xi = torch.masked_select(original_x[mask][:,0], pred_softmax[mask].max(axis=1).indices==k) # Use pred_softmax for assignments
            yi = torch.masked_select(original_x[mask][:,1], pred_softmax[mask].max(axis=1).indices==k) # Use pred_softmax for assignments
            ## hard centroid (calculated from original_x for display, but not used in loss)
            xm=xi.mean()
            ym=yi.mean()
            plt.plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
            plt.scatter(xm,ym, marker='*',c='k',s=30)
            # Centroids from embeddings are high-dimensional, so we cannot plot their first two dimensions
            # directly on the 2D map. Commenting out centroid plotting.
            # plt.scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20)
        plt.show()

    else:
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(15, int(15*aspectratio)))

        for k in range(pred_softmax.shape[1]): # Use pred_softmax here
            xi = torch.masked_select(original_x[mask][:,0], pred_softmax[mask].max(axis=1).indices==k) # Use pred_softmax for assignments
            yi = torch.masked_select(original_x[mask][:,1], pred_softmax[mask].max(axis=1).indices==k) # Use pred_softmax for assignments
            ## hard centroid (calculated from original_x for display, but not used in loss)
            xm=xi.mean()
            ym=yi.mean()
            r = k//4
            c = k%4
            if num_clusters>4:
                axs[r,c].plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
                axs[r,c].scatter(xm,ym, marker='*',c='k',s=30)
                # Centroids from embeddings are high-dimensional, so we cannot plot their first two dimensions
                # directly on the 2D map. Commenting out centroid plotting.
                # axs[r,c].scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20)
                axs[r,c].set_title(str(k)+' ('+str(len(xi))+')')
            else:
                axs[c].plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
                axs[c].scatter(xm,ym, marker='*',c='k',s=30)
                # Centroids from embeddings are high-dimensional, so we cannot plot their first two dimensions
                # directly on the 2D map. Commenting out centroid plotting.
                # axs[c].scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20)
                axs[c].set_title(str(k)+' ('+str(len(xi))+')')

        plt.show()