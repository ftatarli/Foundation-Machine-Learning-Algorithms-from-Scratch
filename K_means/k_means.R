X <- matrix(c(
  1.0,  2.0,
  1.5,  1.8,
  5.0,  8.0,
  8.0,  8.0,
  1.0,  0.6,
  9.0, 11.0
), ncol = 2, byrow = TRUE)

k <- 2
print(X)

## Step 0 — the distance metric
centroids_start <- matrix(c(
  1.0, 2.0,
  5.0, 8.0
), ncol = 2, byrow = TRUE)

centroids_start

euclidean <- function(a, b) {
  total <- 0
  for (d in seq_along(a)) {
    total <- total + (a[d] - b[d])^2
  }
  sqrt(total)
}

# check: rows 1 and 2 -> 0.5385165
euclidean(X[1, ], X[2, ])

## Step 1 — `compute_distances`: every point to every centroid
compute_distances <- function(X, centroids) {
  n <- nrow(X)
  k <- nrow(centroids)
  distances <- matrix(0, nrow = n, ncol = k)

  for (i in seq_len(n)) {          # each point
    for (j in seq_len(k)) {        # each centroid
      distances[i, j] <- euclidean(X[i, ], centroids[j, ])
    }
  }
  distances
}

D <- compute_distances(X, centroids_start)
print(round(D, 3))

## Step 2 — `assign_clusters`: nearest centroid wins
assign_clusters <- function(distances) {
  n <- nrow(distances)
  k <- ncol(distances)
  labels <- integer(n)

  for (i in seq_len(n)) {
    best <- 1                            # assume centroid 1 is closest...
    for (j in seq_len(k)) {              # ...then try to beat it
      if (distances[i, j] < distances[i, best]) best <- j
    }
    labels[i] <- best
  }
  labels
}

labels <- assign_clusters(D)
print(labels)

## Step 3 — `update_centroids`: move each center to its cluster's mean
update_centroids <- function(X, labels, centroids, k) {
  n <- nrow(X)
  p <- ncol(X)
  new_centroids <- matrix(0, nrow = k, ncol = p)

  for (j in seq_len(k)) {
    total <- rep(0, p)
    count <- 0

    for (i in seq_len(n)) {              # collect the points of cluster j
      if (labels[i] == j) {
        total <- total + X[i, ]
        count <- count + 1
      }
    }

    if (count == 0) {
      new_centroids[j, ] <- centroids[j, ]   # empty cluster -> freeze it
    } else {
      new_centroids[j, ] <- total / count    # the mean
    }
  }
  new_centroids
}

centroids_new <- update_centroids(X, labels, centroids_start, k)
print(round(centroids_new, 4))
