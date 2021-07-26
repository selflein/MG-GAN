from mggan.utils import count_parameters
from mggan.model.modules.standard import MultiGenerator
from mggan.model.modules.discriminators import MultiDiscriminatorTrajectory
from mggan.model.modules.standard_discrete import DiscreteLatentGenerator


def construct_model(config):
    """
    :param config: Neural Network Architecture Configurations
    :return:
        G: generator structure
        D: discriminator structure
    """
    unbound_output = config.gan_obj in ["W", "LS"]
    num_discs = 5 if config.gan_type == "probgan" else 1
    config.use_pinet = config.weighting_target != "none" and not config.unconditional

    pred_len = 12
    scene_dim = 8 * 8

    if config.experiment == "multi_generator":
        G = MultiGenerator(
            z_size=config.noise_dim,
            inp_format=config.inp_format,
            encoder_h_dim=config.h_dim,
            decoder_h_dim=config.decoder_h_dim,
            social_feat_size=config.h_dim if config.n_social_modules > 0 else 0,
            embedding_dim=int(config.decoder_h_dim // 2),
            num_gens=config.num_gens,
            pred_len=pred_len,
            pool_type=config.pool_type,
            num_social_modules=config.n_social_modules,
            scene_dim=scene_dim,
            use_pinet=config.use_pinet,
            learn_prior=config.unconditional,
        )

        D = MultiDiscriminatorTrajectory(
            num_discs=num_discs,
            num_gens=config.num_gens,
            unbound_output=unbound_output,
            h_dim=config.h_dim * 2,
            pred_len=pred_len,
            inp_format=config.inp_format,
            gan_type=config.gan_type,
            scene_dim=scene_dim,
            global_disc=config.global_disc,
            pool_type=config.pool_type,
        )
    elif config.experiment == "discrete":
        G = DiscreteLatentGenerator(
            z_size=config.noise_dim,
            inp_format=config.inp_format,
            encoder_h_dim=config.h_dim,
            decoder_h_dim=config.decoder_h_dim,
            social_feat_size=config.h_dim if config.n_social_modules > 0 else 0,
            embedding_dim=16,
            num_gens=config.num_gens,
            pred_len=pred_len,
            pool_type=config.pool_type,
            num_social_modules=config.n_social_modules,
            scene_dim=scene_dim,
            use_pinet=config.use_pinet,
            learn_prior=config.unconditional,
        )

        D = MultiDiscriminatorTrajectory(
            num_discs=num_discs,
            num_gens=config.num_gens,
            unbound_output=unbound_output,
            h_dim=config.h_dim * 2,
            pred_len=pred_len,
            inp_format=config.inp_format,
            gan_type=config.gan_type,
            scene_dim=scene_dim,
            global_disc=config.global_disc,
            pool_type=config.pool_type,
        )

    else:
        raise ValueError("Requested model not implemented.")

    print("G #parameters: ", count_parameters(G))
    print("D #parameters: ", count_parameters(D))
    config.num_gen_parameters = count_parameters(G)
    return G, D
