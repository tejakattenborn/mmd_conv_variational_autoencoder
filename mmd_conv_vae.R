# Representation learning with maximum mean discrepancy convolutional variational autoencoder (MMD-VAE)
# This is a tensorflow 2 compatible version of the original code (tensorflow 1) posted by Sigrid Keydana on the TensorFlow for R blog.
# https://blogs.rstudio.com/tensorflow/posts/2018-10-22-mmd-vae/



library(keras)
library(tensorflow)
library(tfdatasets)
library(dplyr)
library(ggplot2)
library(glue)

# Setup and preprocessing -------------------------------------------------

fashion <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion$train
c(test_images, test_labels) %<-% fashion$test

train_x <-
  train_images %>% `/`(255) %>% k_reshape(c(60000, 28, 28, 1)) %>% k_cast(dtype='float32')
test_x <-
  test_images %>% `/`(255) %>% k_reshape(c(10000, 28, 28, 1)) %>% k_cast(dtype='float32')

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

buffer_size <- 60000
batch_size <- 100
batches_per_epoch <- buffer_size / batch_size

train_dataset <- tensor_slices_dataset(train_x) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

test_dataset <- tensor_slices_dataset(test_x) %>%
  dataset_batch(10000)



# Model -------------------------------------------------------------------

latent_dim <- 2

encoder_model <- function(name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <-
      layer_conv_2d(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        activation = "relu",
        dtype='float32'
      )
    self$conv2 <-
      layer_conv_2d(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        activation = "relu"
      )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = latent_dim)
    
    function (x, mask = NULL) {
      x %>%
        self$conv1() %>%
        self$conv2() %>%
        self$flatten() %>%
        self$dense() #%>%
    }
  })
}

decoder_model <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    self$dense <- layer_dense(units = 7 * 7 * 32, activation = "relu")
    self$reshape <- layer_reshape(target_shape = c(7, 7, 32))
    
    self$deconv1 <-
      layer_conv_2d_transpose(
        filters = 64,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv2 <-
      layer_conv_2d_transpose(
        filters = 32,
        kernel_size = 3,
        strides = 2,
        padding = "same",
        activation = "relu"
      )
    self$deconv3 <-
      layer_conv_2d_transpose(
        filters = 1,
        kernel_size = 3,
        strides = 1,
        padding = "same",
        activation = "sigmoid"
      )
    
    function (x, mask = NULL) {
      x %>%
        self$dense() %>%
        self$reshape() %>%
        self$deconv1() %>%
        self$deconv2() %>%
        self$deconv3()
    }
  })
}


optimizer <- tf$keras$optimizers$Adam(1e-4)


compute_kernel <- function(x, y) {
  x_size <- k_shape(x)[1]
  y_size <- k_shape(y)[1]
  dim <- k_shape(x)[2]
  #tiled_x <- k_tile(k_reshape(x, k_stack(list(x_size, 1, dim))), k_stack(list(1, y_size, 1)))
  #tiled_y <- k_tile(k_reshape(y, k_stack(list(1, y_size, dim))), k_stack(list(x_size, 1, 1)))
  
  tiled_x <- k_tile(k_reshape(x, k_stack(list(x_size, 1L, dim))), k_stack(list(1L, y_size, 1L)))
  tiled_y <- k_tile(k_reshape(y, k_stack(list(1L, y_size, dim))), k_stack(list(x_size, 1L, 1L)))

  
  k_exp(-k_mean(k_square(tiled_x - tiled_y), axis = 3L) / k_cast(dim, tf$float32))
}

compute_mmd <- function(x, y, sigma_sqr = 1) {
  x_kernel <- compute_kernel(x, x)
  y_kernel <- compute_kernel(y, y)
  xy_kernel <- compute_kernel(x, y)
  k_mean(x_kernel) + k_mean(y_kernel) - 2 * k_mean(xy_kernel)
}



# Output utilities --------------------------------------------------------

num_examples_to_generate <- 64

random_vector_for_generation <-
  k_random_normal(shape = list(num_examples_to_generate, latent_dim),
                  dtype = tf$float32)

generate_random_clothes <- function(epoch) {
  predictions <-
    decoder(random_vector_for_generation) %>% tf$nn$sigmoid()
  png(paste0("cvae_clothes_epoch_", epoch, ".png"))
  par(mfcol = c(8, 8))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  for (i in 1:64) {
    img <- predictions[i, , , 1]
    img <- t(apply(img, 2, rev))
    image(
      1:28,
      1:28,
      img * 127.5 + 127.5,
      col = gray((0:255) / 255),
      xaxt = 'n',
      yaxt = 'n'
    )
  }
  dev.off()
}


show_latent_space <- function(epoch) {
  
  iter <- make_iterator_one_shot(test_dataset)
  x <-  iterator_get_next(iter)
  x_test_encoded <- encoder(x)  #check that [[1]] must not be added
  x_test_encoded %>%
    as.matrix() %>%
    as.data.frame() %>%
    mutate(class = class_names[fashion$test$y + 1]) %>%
    ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +
    theme(aspect.ratio = 1) +
    theme(plot.margin = unit(c(0, 0, 0, 0), "null")) +
    theme(panel.spacing = unit(c(0, 0, 0, 0), "null"))
  
  ggsave(
    paste0("mmd_latentspace_epoch_", epoch, ".png"),
    width = 10,
    height = 10,
    units = "cm"
  )
}


show_grid <- function(epoch) {
  png(paste0("mmd_grid_epoch_", epoch, ".png"))
  par(mar = c(0.5, 0.5, 0.5, 0.5),
      xaxs = 'i',
      yaxs = 'i')
  n <- 16
  img_size <- 28
  
  grid_x <- seq(-4, 4, length.out = n)
  grid_y <- seq(-4, 4, length.out = n)
  rows <- NULL
  
  for (i in 1:length(grid_x)) {
    column <- NULL
    for (j in 1:length(grid_y)) {
      z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
      column <-
        rbind(column,
              (decoder(z_sample, 'float32') %>% as.numeric()) %>% matrix(ncol = img_size))
    }
    rows <- cbind(rows, column)
  }
  rows %>% as.raster() %>% plot()
  dev.off()
}


# Training loop -----------------------------------------------------------

num_epochs <- 50

encoder <- encoder_model()
decoder <- decoder_model()

checkpoint_dir <- "./checkpoints_fashion_cvae"
checkpoint_prefix <- file.path(checkpoint_dir, "ckpt")
checkpoint <-
  tf$train$Checkpoint(optimizer = optimizer,
                      encoder = encoder,
                      decoder = decoder)

generate_random_clothes(0)
show_latent_space(0)
show_grid(0)


for (epoch in seq_len(num_epochs)) {
  iter <- make_iterator_one_shot(train_dataset)
  
  total_loss <- 0
  loss_nll_total <- 0
  loss_mmd_total <- 0
  
  until_out_of_range({
    x <-  iterator_get_next(iter)
    

    with(tf$GradientTape(persistent = TRUE) %as% tape, {
      
      mean <- encoder(x)
      preds <- decoder(mean)
      
      true_samples <- k_random_normal(shape = c(batch_size, latent_dim), dtype = tf$float32)
      loss_mmd <- compute_mmd(true_samples, mean)
      loss_nll <- k_mean(k_square(x - preds))
      loss <- loss_nll + loss_mmd
      
    })
    
    total_loss <- total_loss + loss
    loss_mmd_total <- loss_mmd + loss_mmd_total
    loss_nll_total <- loss_nll + loss_nll_total
    
    encoder_gradients <- tape$gradient(loss, encoder$variables)
    decoder_gradients <- tape$gradient(loss, decoder$variables)
    
    optimizer$apply_gradients(purrr::transpose(list(
      encoder_gradients, encoder$variables
    )))
    optimizer$apply_gradients(purrr::transpose(list(
      decoder_gradients, decoder$variables
    )))

  
  })
  
  checkpoint$save(file_prefix = checkpoint_prefix)
  

  cat(
    glue(
      "Losses (epoch): {epoch}:",
      "  {(as.numeric(loss_nll_total)/batches_per_epoch) %>% round(4)} loss_nll_total,",
      "  {(as.numeric(loss_mmd_total)/batches_per_epoch) %>% round(4)} loss_mmd_total,",
      "  {(as.numeric(total_loss)/batches_per_epoch) %>% round(4)} total"
    ),
    "\n"
  )
  
  if (epoch %% 10 == 0) {
    generate_random_clothes(epoch)
    show_latent_space(epoch)
    show_grid(epoch)
  }
  print(paste0("finished epoch ", epoch))
}
