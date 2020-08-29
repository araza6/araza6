---

title: "Halfway through Google Summer of Code"

date: "2020-07-19"

description: "Google Summer of Code 2020 updates"

tags: [gsoc]


---

It is about mid-July, and I am midway through my internship with Google Summer of Code (GSoC). This post is intended to serve as a quick update on the progress of my ongoing project about autodifferentiation of functions involving quantum processes.

In my [last post](https://araza6.github.io/posts/qobj-jax-interface/), I highlighted how is it difficult to interface QuTiP with JAX. As a (temporary) solution, I highlighted towards the end of the post, that the way forward would be to make a light-weight, but JAX-compatible, QuTiP. Two weeks later, here we are with a light-weight QuTiP that mimics essential quantum operations available in QuTiP. I expect that our initial set of users would be the ones who might have used QuTiP in the past. With that in mind, the API is as close to QuTiP as possible. All the QuTiP functions that I have rewritten, I have tried my best to keep the arguments, and function names the same. While the light-weight QuTiP is still very light, in the sense that it has a very limited set of functions, I will continue expanding upon the functions as we move on. 

The functions that are currently supported, however, are sufficient to reproduce some papers that can already show all the amazing things we can do with `qgrad_qutip`. At the moment, I am working on reproducing [this](https://arxiv.org/abs/1901.03431), [this](https://arxiv.org/abs/2001.11897) and [this](https://arxiv.org/abs/2004.14256) paper. 
The pull requests for these are still under works since JAX's autodiff comes with some caveats. More on these caveats in a another post.

Looking ahead, I plan to wrap up the documentation and tutorials for later half of July.

