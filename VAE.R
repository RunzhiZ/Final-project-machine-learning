if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()
library(keras)
library(ggplot2)
library(dplyr)
K <- keras::backend()
setwd('/blue/datta/runzhi.zhang/Machine learning/')
# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 256L
epochs <- 50L
epsilon_std <- 1.0

# Model definition --------------------------------------------------------

x <- layer_input(shape = c(original_dim))
h <- layer_dense(x, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean=0.,
    stddev=epsilon_std
  )
  z_mean + k_exp(z_log_var/2)*epsilon
}
# note that "output_shape" isn't necessary with the TensorFlow backend
z <- layer_concatenate(list(z_mean, z_log_var)) %>%
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
h_decoded <- decoder_h(z)
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_h(decoder_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2)


vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}



# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")
set.seed(32608)
train_index<-sample(60000,5000,replace=F)
test_index<-sample(10000,1000,replace=F)
x_train_1<-x_train[train_index,]
x_test_1<-x_test[test_index,]


vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% fit(
  x_train_1, x_train_1,
  shuffle = TRUE,
  epochs = 10,
  batch_size = batch_size,
  validation_data = list(x_test_1, x_test_1)
)

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}

png(file="VAE_10.png")
rows %>% as.raster() %>% plot()
dev.off()

#### epochs = 50 #### 
vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% fit(
  x_train_1, x_train_1,
  shuffle = TRUE,
  epochs = 50,
  batch_size = batch_size,
  validation_data = list(x_test_1, x_test_1)
)
rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
png(file="VAE_50.png")
rows %>% as.raster() %>% plot()
dev.off()

#### epochs = 200 #### 
vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% fit(
  x_train_1, x_train_1,
  shuffle = TRUE,
  epochs = 200,
  batch_size = batch_size,
  validation_data = list(x_test_1, x_test_1)
)
rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
png(file="VAE_200.png")
rows %>% as.raster() %>% plot()
dev.off()

#### epochs = 1000 #### 
vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% fit(
  x_train_1, x_train_1,
  shuffle = TRUE,
  epochs = 1000,
  batch_size = batch_size,
  validation_data = list(x_test_1, x_test_1)
)
rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
png(file="VAE_1000.png")
rows %>% as.raster() %>% plot()
dev.off()