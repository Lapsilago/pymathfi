import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class GMM_Conditional_Distribution:

  '''
  This class takes as input the means, covariances and weights of all the 
  components of a GMM p(Y,X) and outputs the conditional mean and conditional 
  variance/covariance of the GMM p(Y|X) for an input vector/matrix X, as
  well as the new component weights and the beta = "Cov(y1,y2|X)/Var(y1|X)"

  Refer to the paper "GMM DCKE - Semi-Analytic Conditional Expectations"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3902490"


  Dependencies:
 
  Requires Tensorflow and Tensorflow Probability. If Tensorflow is already installed, then Tensorflow Probability can 
  be installed in your Python Environment by using the command "pip install --upgrade tensorflow-probability"

  Instructions:
  Suppose, for e.g. a Gaussian Mixture Model for the data (Y1, Y2, Y3, X1, X2, X3) is available. The parameters for conditional distribution
  P (Y1, Y2, Y3| X1, X2, X3) can be obtained using this class.

  1) The class object has to be initialized with the input: 
     i) gmm_means - Means of the GMM components 
     ii) gmm_covariances - Covariances of the GMM components
     iii) gmm_weights - Weight for each GMM component
     iv) size_dim_Y - Dimension of the variables Y that are to be conditioned. In the above e.g, size_dim_Y = 3.
  
  2) Use the class method "conditional_gmm_parameters". It takes as input a vector/matrix X with the shape (number_of_samples, dimensions_X)
     For all samples of input X, it outputs the corresponding in this order:
      i) Conditional Expectation E(Y | X = x) with the shape (number_of_samples, number_dimensions_Y). In the above e.g., the shape will be (n_samples, 3)
      ii) Conditional Covariance Matrix Cov(Y | X = x) with the shape (number_of_samples, n_dimensions_Y, n_dimensions_Y). In the above e.g., the shape will be (n_samples, 3, 3)
      iii) New weights with the shape (number_of_samples, number_of_mixture_components)
  
  3) To calculate the Beta, use the class method "beta". It takes as input
     i) index_Z : the index of the control variate Z (from the paper)
     ii) index_Y : the index of the main variable of interest 
     In the above example, if Y2 is a control variate and Y3 is the variable for which conditional expecation is being computed
     then index_Z = 1, and index_Y = 2  

     The function will output a beta that corresponds to each sample of Z, with the shape (number_of_samples)

     Note that in case of multiple control variates, the beta for each control variate should be computed separately. 


  




  '''

  def __init__(self, gmm_means, gmm_covariances, gmm_weights, size_dim_Y = 1):
      """ 
          gmm_means - list of means of mixture components - shape (n_gmm_components, dim_mixture_distribution)
          gmm_covariances - list of covariance matrices of mixture components - shape (n_gmm_components, dim_mixture_dist, dim_mixture_dist)
          gmm_weights - list of weights assigned to each mixture component - shape (n_gmm_components)
          size_dim_Y - dimension of Y for Y|X (by default = 1) - see instructions

      """

      
      self.n_components = len(gmm_weights)
      self.cluster_weights = np.array(gmm_weights).reshape(self.n_components, 1)

      self.size_dim_Y = size_dim_Y
      # self.index_map = {x:i for i,x in enumerate(self.var_list_Y)}
      self.cond_mu = None
      self.cond_cov = None
      self.beta_final = None
      self.new_weights = None
      self.unweighted_mu = None
      self.unweighted_cov = None
      if gmm_means.shape[1] < self.size_dim_Y:
        raise Exception("Dimensions don't match")
      
      # separate Y and X for E(Y|X=x) and Var(Y|X=x) 
      self.mu_Y = []
      self.mu_X = []
      self.sigma_Y = []
      self.sigma_YX = []
      self.sigma_XY = []
      self.sigma_X = []

      
      for i in range(self.n_components):
        
        self.mu_Y.append(gmm_means[i , 0:self.size_dim_Y].reshape(-1,1)) 
        self.mu_X.append(gmm_means[i, self.size_dim_Y:].reshape(-1,1))

        # list slicing -  separate the larger covariance matrix into submatrices according to dimensions
        
        self.sigma_Y.append(gmm_covariances[i, 0:self.size_dim_Y, 0:self.size_dim_Y])
        self.sigma_YX.append(gmm_covariances[i, 0:self.size_dim_Y, self.size_dim_Y:])
        self.sigma_XY.append(gmm_covariances[i, self.size_dim_Y:, 0:self.size_dim_Y])
        self.sigma_X.append(gmm_covariances[i, self.size_dim_Y:, self.size_dim_Y:])


  def _local_class_variables(self, X):
    self.unweighted_mu = []
    self.unweighted_cov = []
    if len(X.shape) == 1:
      X = X.reshape(-1,1)
    
    for i in range(self.n_components):

      A = (X.transpose() - self.mu_X[i])
      B = tf.linalg.matmul(self.sigma_YX[i], tf.linalg.inv(self.sigma_X[i]))
      self.unweighted_mu.append(tf.add(self.mu_Y[i], tf.linalg.matmul(B, A)).numpy())

      self.unweighted_cov.append(self.sigma_Y[i] - tf.linalg.matmul(B, self.sigma_XY[i])) 
      
    return self.unweighted_mu, self.unweighted_cov # self.unweighted_mu shape (number_gmm_components, size_dim_Y, number_samples) 
                                # self.unweighted_cov shape (number_gmm_components, size_dim_Y, size_dim_Y)

         
  def conditional_weights(self, X):

    '''
    Input: X shape (number_of_samples, dimensions_X)

    Output: new sample weights shape (number_of_samples, number_of_mixture_components)

    '''
    if len(X.shape) == 1:
      X = X.reshape(-1,1)

    self.new_weights = np.empty(shape=(self.n_components, X.shape[0]))
    for i in range(self.n_components):
      likelihood =  tfp.distributions.MultivariateNormalTriL(loc=self.mu_X[i].flatten(), scale_tril=tf.linalg.cholesky(self.sigma_X[i]))

      self.new_weights[i, :] = likelihood.prob(X).numpy()

    self.new_weights = self.cluster_weights*self.new_weights
    self.new_weights = (self.new_weights/np.sum(self.new_weights, axis = 0)).T

    return self.new_weights



  def conditional_expectation(self, X):

    '''
    input: X shape (number_of_samples, dimensions_X)
    output: E(Y|X) shape (number_of_samples, number_dimensions_Y)
    '''
    if len(X.shape) == 1:
      X = X.reshape(-1,1)

    if self.new_weights is None:
      self.new_weights = self.conditional_weights(X)
    
    if self.unweighted_mu is None or self.unweighted_cov is None:
      self.unweighted_mu, self.unweighted_cov = self._local_class_variables(X)
    
    unweighted_means = np.asarray(self.unweighted_mu).T.swapaxes(1,2)

    weighted_mixture_means = tfp.distributions.MixtureSameFamily(
                                    mixture_distribution=tfp.distributions.Categorical(probs=self.new_weights),
                                    components_distribution=tfp.distributions.MultivariateNormalTriL(loc=unweighted_means, \
                                      scale_tril=tf.linalg.cholesky(self.unweighted_cov)))    
    self.cond_mu = weighted_mixture_means.mean().numpy()    

    return self.cond_mu
    
  def conditional_covariance(self, X):
    '''
    Input: X shape (number_of_samples, dimensions_X)
    Output: Cov(Y|X) shape (number_of_samples, n_dimensions_Y, n_dimensions_Y)
    
    '''
    if len(X.shape) == 1:
      X = X.reshape(-1,1)


    if self.new_weights is None:
      self.new_weights = self.conditional_weights(X)
    
    if self.unweighted_cov is None:
      _ , self.unweighted_cov = self._local_class_variables(X)
    
    weighted_mixture_covariances = tfp.distributions.MixtureSameFamily(
                                mixture_distribution=tfp.distributions.Categorical(probs=self.new_weights),
                                components_distribution=tfp.distributions.MultivariateNormalTriL(
                                                                          loc= np.zeros((1, self.n_components, \
                                                                          self.size_dim_Y)),\
                                                                            scale_tril=tf.linalg.cholesky(np.asarray(self.unweighted_cov))))

    self.cond_cov = weighted_mixture_covariances.covariance().numpy()

    return self.cond_cov

  def beta(self, index_Z, index_Y):

    '''
    Input: index_Z - index of control variate
           index_Y - index of variable whose conditional expectation is being computed  
    
    Output: Beta shape (number_of_samples)
    
    '''


    if self.cond_cov is None:
        raise Exception\
        ("Conditional Covariance matrices have not been computed. Please run the \"conditional_gmm_parameters\" function before calculating the beta")

    self.beta_final = self.cond_cov[:, index_Z, index_Y]/self.cond_cov[:, index_Z, index_Z]
    return self.beta_final
    


  def conditional_gmm_parameters(self, X):
    '''
      Input X shape (number_samples_per_dimension, dimensions_X)
      
      Output:
      Conditional Expectation/Weighted Mean E(Y|X=x) shape (number_of_samples, number_dimensions_Y)
      Weighted Conditional Covariance Matrix = Cov(Y|X=x) shape (number_of_samples, n_dimensions_Y, n_dimensions_Y)
      New weights for Conditional GMM shape (number_of_samples, number_of_mixture_components)

    '''

    if self.new_weights is None:
      self.new_weights = self.conditional_weights(X)

    if self.cond_mu is None:
      self.cond_mu = self.conditional_expectation(X)

    if self.cond_cov is None:
      self.cond_cov = self.conditional_covariance(X)




    return self.cond_mu, self.cond_cov, self.new_weights