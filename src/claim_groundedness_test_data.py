# ==================================================
# CLAIM-LEVEL GROUNDEDNESS TEST DATA
# ==================================================
#
# Each case contains:
#
#   claim   : generated claim
#   context : available evidence
#   grounded: ground-truth label
#
# ==================================================


CLAIM_GROUNDEDNESS_TEST_CASES = [

    # ==================================================
    # SUPPORTED CLAIMS
    # ==================================================

    {
        "claim":
            "The Turing test was invented in 1950 by Alan Turing.",

        "context":
            "The concept of Artificial Intelligence traces back "
            "to as early as 1950 when Alan Turing invented the "
            "Turing test.",

        "grounded": True
    },


    {
        "claim":
            "Siri was announced as a digital assistant by Apple in 2011.",

        "context":
            "In 2011, Siri was announced as a digital assistant "
            "by Apple.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning gives computers the ability to learn "
            "without being explicitly programmed.",

        "context":
            "Machine learning is the discipline that gives "
            "computers the ability to learn without being "
            "explicitly programmed.",

        "grounded": True
    },


    {
        "claim":
            "Artificial Intelligence helps machines solve complex "
            "problems using humanlike intelligence.",

        "context":
            "Artificial Intelligence is a computing concept "
            "that helps a machine think and solve complex "
            "problems as we humans do with our intelligence.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning can be used to create balanced gameplay.",

        "context":
            "Machine learning is used in gaming to create "
            "the most balanced gameplay possible.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning is used in gaming to make games "
            "more interesting.",

        "context":
            "Game designers can use machine learning to make "
            "their games more interesting.",

        "grounded": True
    },


    {
        "claim":
            "OpenAI was founded in 2015.",

        "context":
            "Elon Musk and some others founded OpenAI in 2015.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning is used for fraud detection.",

        "context":
            "Machine learning is used in various sectors for "
            "different reasons, including investing, advertising, "
            "marketing, ecommerce, banking, organizing news, "
            "fraud detection, and more.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning can be used to predict outcomes.",

        "context":
            "Machine learning uses algorithms trained on data "
            "to create models capable of predicting outcomes.",

        "grounded": True
    },


    {
        "claim":
            "Machine learning can classify information.",

        "context":
            "Machine learning models are capable of predicting "
            "outcomes and classifying information.",

        "grounded": True
    },


    # ==================================================
    # UNSUPPORTED / CONTRADICTED CLAIMS
    # ==================================================

    {
        "claim":
            "Siri was announced by Apple in 2008.",

        "context":
            "In 2011, Siri was announced as a digital assistant "
            "by Apple.",

        "grounded": False
    },


    {
        "claim":
            "OpenAI was founded by Bill Gates in 2015.",

        "context":
            "Elon Musk and some others founded OpenAI in 2015.",

        "grounded": False
    },


    {
        "claim":
            "OpenAI was founded in 2012.",

        "context":
            "Elon Musk and some others founded OpenAI in 2015.",

        "grounded": False
    },


    {
        "claim":
            "The Turing test was invented by Albert Einstein.",

        "context":
            "Alan Turing invented the Turing test in 1950.",

        "grounded": False
    },


    {
        "claim":
            "Machine learning completely replaces human decision-making.",

        "context":
            "Machine learning is used to make predictions, "
            "classify information, and automate operations.",

        "grounded": False
    },


    {
        "claim":
            "Machine learning predicts outcomes with perfect accuracy.",

        "context":
            "Machine learning models can be used to predict "
            "outcomes.",

        "grounded": False
    },


    {
        "claim":
            "Machine learning is only used in gaming.",

        "context":
            "Machine learning is used in search engines, "
            "digital cameras, credit card transactions, "
            "fraud detection, and gaming.",

        "grounded": False
    },


    {
        "claim":
            "Siri was created by Google.",

        "context":
            "In 2011, Siri was announced as a digital assistant "
            "by Apple.",

        "grounded": False
    },


    {
        "claim":
            "Alan Turing invented the Turing test at Oxford University.",

        "context":
            "Alan Turing invented the Turing test in 1950.",

        "grounded": False
    },


    {
        "claim":
            "Machine learning can guarantee that predictions "
            "will always be correct.",

        "context":
            "Machine learning improves predictions using "
            "algorithms and past data.",

        "grounded": False
    }
]