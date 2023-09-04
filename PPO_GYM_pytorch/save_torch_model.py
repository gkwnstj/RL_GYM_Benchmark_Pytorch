import torch

def saveasONNX(actor_model, critic_model, gpudevice, reporter, episode, onnx_path, episode_reward):
    onnx_name_actor = onnx_path + "actor"+ str(episode)  + ".onnx"
    onnx_name_critic = onnx_path + "critic"+ str(episode)  + ".onnx"
    actor_model.eval()
    critic_model.eval()

    dummy_state = torch.empty(1, 3, device=gpudevice, requires_grad=True)
    dummy_action = torch.randn(1, 1, device=gpudevice, requires_grad=True)        
    torch.onnx.export(
        actor_model, 
        dummy_state, 
        onnx_name_actor,
        verbose=False,
        input_names= ['input'],
        output_names = ['output'],
        )
    # torch.onnx.export(
    #     critic_model, 
    #     [dummy_state, dummy_action], 
    #     onnx_name_critic,
    #     verbose=False
    #     )  
    reporter.info(f"===== ONNX SAVED at {onnx_name_actor}=== reward {episode_reward}")