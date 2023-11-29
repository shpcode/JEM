


class Sampler:

    def __init__(self, model,mode,  img_shape, sample_size,from_scrach, max_len=8192):
        """
        This code is an adaptation of the Langevin sampler from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
        The mode feature is added to allow sampling the joint distribution.
        """
        super().__init__()
        self.model = model
        self.soft = torch.nn.Softmax(dim=1)
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]   ## do I need to do same thing for labels!!
        self.from_scrach = from_scrach
        self.mode = mode

    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        for i in range(100):
          n_new=np.random.binomial(self.sample_size, self.from_scrach)
          if n_new !=0:
            break

        rand_imgs = (torch.rand((n_new,) + self.img_shape) * 2 - 1).to(device)
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0).to(device)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach()

        if self.mode=='joint':
          rand_labels = (torch.multinomial(torch.tensor(labels).float(), n_new, replacement=True) ).to(device)
          old_labels = torch.argmax(self.soft(self.model.logits(old_imgs)) , 1)
          inp_labs = torch.cat([rand_labels, old_labels], dim = 0 ).detach()
        else:
          inp_labs = None

        # Perform MCMC sampling
        inp_imgs , inp_labs = Sampler.generate_samples(self.model, inp_imgs, inp_labs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs , inp_labs

    @staticmethod
    def generate_samples(model, inp_imgs, inp_labs, steps=60, step_size=10, return_img_per_step=False):
        soft = torch.nn.Softmax(dim=1)
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True


        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for step in range(steps):


            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)


            # sample y
            if inp_labs !=None and step < 5:
              p = soft(model.logits(inp_imgs.clone().detach()))
              inp_labs = torch.multinomial(p, 1, replacement=False)

            # Part 2: calculate gradients for the current input.
            if inp_labs == None:
              out_imgs = -model(inp_imgs)
            else:
              out_imgs = -model.logits(inp_imgs).gather(1,inp_labs.view(-1,1))

            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs , inp_labs
