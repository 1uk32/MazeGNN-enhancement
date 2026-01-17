import torch

def kmeansloss(cluster_assignment, coords):
    loss_d=0
    loss_d_neg=0
    loss_invd_neg=0
    loss_inter=0

    # Ensure cluster_assignment has finite values
    cluster_assignment = torch.nan_to_num(cluster_assignment, nan=0.0, posinf=0.0, neginf=0.0)

    cluster_size = torch.sum(cluster_assignment,axis=0).reshape(cluster_assignment.shape[1],1)

    # Handle empty clusters by adding a small epsilon to cluster_size before division
    # This prevents NaN in centroids if a cluster_size is 0
    safe_cluster_size = cluster_size.clone()
    safe_cluster_size[safe_cluster_size == 0] = 1e-6 # Set to a small non-zero value

    centroids = torch.div(torch.matmul(cluster_assignment.t(),coords), safe_cluster_size)

    num_nodes = cluster_assignment.shape[0]
    num_clusters = cluster_assignment.shape[1]

    for k in range(num_clusters):
        # Select D-dimensional coordinates for nodes within the current cluster
        cluster_mask_in = (cluster_assignment.max(axis=1).indices==k)
        coords_in_cluster = coords[cluster_mask_in]

        # Select D-dimensional coordinates for nodes outside the current cluster
        cluster_mask_out = (cluster_assignment.max(axis=1).indices!=k)
        coords_out_cluster = coords[cluster_mask_out]

        # Calculate distances within the cluster
        if len(coords_in_cluster) > 0:
            dist_within = torch.sum((coords_in_cluster - centroids[k])**2, axis=-1)
            loss_mean_dist2 = torch.mean(dist_within)
            loss_d += loss_mean_dist2
        else:
            loss_mean_dist2 = 0.0 # If cluster is empty, contribution to loss is 0

        # Calculate distances for nodes outside the cluster (negative samples)
        if len(coords_out_cluster) > 0:
            dist_neg = torch.sum((coords_out_cluster - centroids[k])**2, axis=-1)
            loss_mean_dist2_neg = torch.mean(dist_neg)
            loss_mean_inv_d_neg = torch.mean(1 / (dist_neg + 0.01)) # Add epsilon for stability
            loss_d_neg += loss_mean_dist2_neg
            loss_invd_neg += loss_mean_inv_d_neg
        else:
            loss_mean_dist2_neg = 0.0 # If all nodes are in cluster k, no negative samples
            loss_mean_inv_d_neg = 0.0

        # Calculate inter-point distance within the cluster to penalize spread
        if len(coords_in_cluster) > 1: # Need at least 2 points for inter-distance
            # Sample up to 1000 points if the cluster is large, otherwise use all
            if len(coords_in_cluster) < 1000:
                inds = torch.arange(len(coords_in_cluster))
            else:
                inds = torch.randperm(len(coords_in_cluster))[:1000]

            sampled_coords = coords_in_cluster[inds]

            # Calculate pairwise squared Euclidean distance for D-dimensional vectors
            # Reshape for broadcasting: (M, 1, D) - (1, M, D) -> (M, M, D)
            diff_vectors = sampled_coords.unsqueeze(1) - sampled_coords.unsqueeze(0)
            pairwise_sq_distances = torch.sum(diff_vectors**2, dim=-1) # Sum squared differences over D dimensions -> (M, M)
            loss_inter += torch.mean(pairwise_sq_distances)
        else:
            loss_inter += 0.0 # No inter-distance if only one or zero points

    # Penalize when assignment probabilities are uniform (discourage uncertainty)
    reg_assn = -torch.mean(torch.sum(torch.abs(num_clusters * cluster_assignment - 1), axis=-1))

    # Penalize when a cluster has too few nodes (discourage empty clusters)
    # Use safe_cluster_size to avoid division by zero
    reg_num = num_nodes * torch.sum(1 / (safe_cluster_size.squeeze()))

    loss = loss_d - loss_d_neg + 0.1 * loss_invd_neg + 0.1 * reg_assn + 1.5 * reg_num + 0.5 * loss_inter
    print(f'loss: {loss.item():.4f}, d_pos: {loss_d.item():.4f}, d_neg: {loss_d_neg.item():.4f}, loss_invd_neg: {loss_invd_neg.item():.4f}, reg_assn: {reg_assn.item():.4f}, reg_num: {reg_num.item():.4f}, loss_inter: {loss_inter.item():.4f}')
    return loss