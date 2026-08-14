# ==================================================
# GROUNDEDNESS TEST DATA
# ==================================================
#
# Each item contains:
#
#   answer      : sentence we want to evaluate
#   context     : retrieved/source context
#   grounded    : ground-truth label
#
# grounded = True
#     The answer is supported by the context.
#
# grounded = False
#     The answer is NOT supported by the context.
#
# ==================================================


GROUNDEDNESS_TEST_CASES = [

    # ------------------------------------------------
    # 1. Directly supported
    # ------------------------------------------------

    {
        "answer":
            "The Turing test was invented in 1950 by Alan Turing.",

        "context":
            "The concept of Artificial Intelligence traces back "
            "to as early as 1950 when Alan Turing invented the "
            "Turing test.",

        "grounded": True
    },


    # ------------------------------------------------
    # 2. Directly supported
    # ------------------------------------------------

    {
        "answer":
            "Siri was announced as a digital assistant by Apple in 2011.",

        "context":
            "In 2011, Siri was announced as a digital assistant "
            "by Apple.",

        "grounded": True
    },


    # ------------------------------------------------
    # 3. Directly supported
    # ------------------------------------------------

    {
        "answer":
            "Machine learning allows computers to learn without "
            "being explicitly programmed.",

        "context":
            "Machine learning is the discipline that gives "
            "computers the ability to learn without being "
            "explicitly programmed.",

        "grounded": True
    },


    # ------------------------------------------------
    # 4. Paraphrased but supported
    # ------------------------------------------------

    {
        "answer":
            "AI helps machines solve complex problems in ways "
            "similar to human intelligence.",

        "context":
            "Artificial Intelligence is a computing concept "
            "that helps a machine think and solve complex "
            "problems as we humans do with our intelligence.",

        "grounded": True
    },


    # ------------------------------------------------
    # 5. Supported by equivalent wording
    # ------------------------------------------------

    {
        "answer":
            "Machine learning can be used to create more "
            "balanced and interesting games.",

        "context":
            "Machine learning is used in gaming to create "
            "the most balanced gameplay possible. Game "
            "designers can use machine learning to make "
            "their games more interesting.",

        "grounded": True
    },


    # ------------------------------------------------
    # 6. Unsupported date
    # ------------------------------------------------

    {
        "answer":
            "Siri was announced by Apple in 2008.",

        "context":
            "In 2011, Siri was announced as a digital assistant "
            "by Apple.",

        "grounded": False
    },


    # ------------------------------------------------
    # 7. Unsupported person
    # ------------------------------------------------

    {
        "answer":
            "OpenAI was founded in 2015 by Bill Gates.",

        "context":
            "Elon Musk and some others founded OpenAI in 2015.",

        "grounded": False
    },


    # ------------------------------------------------
    # 8. Unsupported claim
    # ------------------------------------------------

    {
        "answer":
            "Machine learning completely replaces human "
            "decision-making in all industries.",

        "context":
            "Machine learning is used in a wide range of "
            "application domains, including search engines, "
            "digital cameras, credit card transactions, "
            "and accident prevention systems in cars.",

        "grounded": False
    },


    # ------------------------------------------------
    # 9. Unsupported location
    # ------------------------------------------------

    {
        "answer":
            "Alan Turing invented the Turing test while "
            "working at Oxford University.",

        "context":
            "The concept of Artificial Intelligence traces "
            "back to as early as 1950 when Alan Turing "
            "invented the Turing test.",

        "grounded": False
    },


    # ------------------------------------------------
    # 10. Unsupported capability
    # ------------------------------------------------

    {
        "answer":
            "Machine learning can predict the future with "
            "perfect accuracy.",

        "context":
            "Machine learning involves computers learning "
            "to improve their predictions using algorithms.",

        "grounded": False
    }
]