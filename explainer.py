


def usegptj(input):
    from transformers import pipeline, set_seed
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    #text = "Q: First list of words is: bad, dark, grim, apocalypse, hate. Second list of words is: bright, good, nice, sunny. Describe the difference between the two lists. A: The first list is filled with positive words the second with negative ones. Q: First list of words is:  sun, water, surfing, ocean, fish . Second list of words is:  climbing, hiking, rocks, trees, goats, snow. Describe the difference between the two lists. A:"
    generator = pipeline('text-generation', model = model, tokenizer = tokenizer)
    gentext = generator(input,max_length=len(input)/4 + 100, num_return_sequences=1)
    print(gentext)
    
if __name__ == '__main__': 
    input = "Task: \n Following is a list of words. \n Please provide a short summary of the list. \n momentsive, fascinating, while, family, hard, original, clever, tale, under, ever, films, screen, also, them, could, interesting, action, nothingble, still, script, worth, plot, should, dialogue, would, feel, down, compelling, everyless, they, audienceity, look, cast, watch, first, character, acting, other, kinder, performanceable, better, both, many, through, entertaininges, director, real, without, such, really, movies, only, sense, humor, there, something, drama, performancesful, over, life, never, their, great, been, made, make, enough, work, which, makes, love, heart, even, comedyd, time, your, what, funny, will, does, little, some, best, much, justness, very, characters, well, into, story, have, about, likey, from, oring, good, most, than, more, movie, this, filmly, itss, with, that \n Answer:"
    usegptj(input)