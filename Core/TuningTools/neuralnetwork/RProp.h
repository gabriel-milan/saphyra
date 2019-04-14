#ifndef TUNINGTOOLS_RPROP_H
#define TUNINGTOOLS_RPROP_H

#include "TuningTools/system/defines.h"

#include <vector>

#include "Gaugi/MsgStream.h"
#include "TuningTools/neuralnetwork/Backpropagation.h"
#include "TuningTools/neuralnetwork/NetConfHolder.h"

namespace TuningTool
{

/** 
 * This class implements the resilient backpropagation training algorithm.
 * This algorithm is based only in the direction of the derivative. The
 * weights are updated by an adaptive value, so this class does not need an
 * update factor.
 *
 * The class can perform either online and batch training, since the instant
 * gradients are automatically accumulated every time the new weights values
 * are calculated and also the class keeps control of how many inputs have
 * been applied to the network, so the instant gradient is used at the update
 * weights phase if only one input was applied, and the mean gradient will be
 * used if multiple inputs were presented to the network. The class also
 * automatically resets the accumulated values after an epoch, preparing
 * itself for the next epoch, so the user just have to use the methods,
 * without worring about internal control.
 **/
class RProp : public Backpropagation
{
  protected:
    //Class attributes.

    /**
     * @brief The maximum allowed learning rate value.
     * Since the upate value can be increased or decreased, this value
     * specifies the maximum accepted value for the update factor.
     **/
    REAL deltaMax;

    /**
     * @brief The minimum allowed learning rate value.
     * Since the upate value can be increased or decreased, this value
     * specifies the minimum accepted value for the update factor.
     **/
    REAL deltaMin;

    /**
     * @brief Specifies the increase factor for the learning rate.
     *
     * If the learning rate must be increased, this factor specifies by how
     * much the learning rate will be increased (the learning rate value is
     * multiplyed by this attribute value).
     **/
    REAL incEta;
    
    /**
     * @brief Specifies the decrease factor for the learning rate.
     *
     * If the learning rate must be decreased, this factor specifies by how
     * much the learning rate will be decreased (the learning rate value is
     * multiplyed by this attribute value).
     **/
    REAL decEta;

    /**
     * @brief The initial learning rate value.
     * This attribute stores the initial learning rate value. After the
     * begining of the training, the current learning rate will be changed
     * according to the Resilient Backpropagation algorithm.
     **/
    REAL initEta;

    /**
     * @brief Stores the delta weights values of the previous training epoch.
     *
     * Since the RProp algorithm must know the previous delta weight values,
     * in order to determine if the learning rate must be increased or
     * decreased, this pointer holds a copy of the delta weights values
     * calculated in the last epoch.
     **/
    REAL ***prev_dw;

    /**
     * @brief Stores the delta biases values of the previous training epoch.
     *
     * Since the RProp algorithm must know the previous delta biases values,
     * in order to determine if the learning rate must be increased or
     * decreased, this pointer holds a copy of the delta biases values
     * calculated in the last epoch.
     **/
    REAL **prev_db;
    
    
    /**
     * @brief The learning rate value for each weight.
     *
     * The speed of this algorithm relies on the fact that it can have an
     * specific learning rate value for each weight. So, this pointer
     * contains the learning rate values that will be used in each weight.
     **/
    REAL ***delta_w;
    
    /**
     * @brief The learning rate value for each bias.
     *
     * The speed of this algorithm relies on the fact that it can have an
     * specific learning rate value for each bias. So, this pointer contains
     * the learning rate values that will be used in each bias.
     **/
    REAL **delta_b;

    /**
     * @brief Gets the smaller of two numbers.
     * This method takes two numbers and returns the smallest of them.
     * @param[in] v1 The first number.
     * @param[in] v2 The second number.
     * @return v1 if v1 < v2, v2 otherwise.
     **/
    REAL min(REAL v1, REAL v2) const {return ((v1 < v2) ? v1 : v2);}
  
    
    /**
     * @brief Gets the largest of two numbers.
     * This method takes two numbers and returns the biggest of them.
     * @param[in] v1 The first number.
     * @param[in] v2 The second number.
     * @return v1 if v1 > v2, v2 otherwise.
     **/
    REAL max(REAL v1, REAL v2) const {return ((v1 > v2) ? v1 : v2);}
    
    /**
     * @brief Gets the sign of a number.
     * This function gets the sign of a number.
     * @param[in] val The number which the sign we want to know.
     * @return 1 if val > 0, -1 if val <0, 0 if val = 0.
     **/
    REAL sign(REAL val) const { if (val > 0) return 1; else if (val < 0) return -1; else return 0;}

    /**
     * @brief Returns if space needed is already allocated
     *
     * Override FeedForward version, with same behavior, but for space needed
     * by RProp.
     **/
    bool isAllocated() const;

    /**
     * @brief Dynamically allocates all the memory we need.
     *
     * This function will take the nNodes vector ans will allocate all the
     * memory that must be dynamically allocated.
     **/
    virtual void allocateSpace();

    /**
     * @brief Applies the RProp update weight algorithm.
     *
     * This method updates each weight and bias by applying the RProp
     * algorithm for weights update. It also saves the current weight value
     * for usage in the next epoch. It also resets the weight and biases
     * values to zero, so they are ready to be used in the next epoch. This
     * method works in a single weight or bias value. So, it will be called
     * as many times as the number of weights and bias in the network.
     *
     * @param delta The learning rate value for the weight or bias.
     * @param d The current delta weight (or bias) value.
     * @param prev_d The previous delta weight (or bias) value.
     * @param w The weight (or bias) value.
     **/
    void updateW(REAL &delta, REAL &d, REAL &prev_d, REAL &w);


    void copyPrevDeltas(const RProp &net);

  public:

    /**
     * @brief Update the weights and bias matrices.
     * Update the bias and weight matrices. It uses the mean
     * gradient sign. It is also prepared to work with nodes activation
     * and frozen nodes.
     * @param[in] numEvents The number of events applied to the network during the training phase.
     * @see TuningTool::Backpropagation#updateWeights()
     **/
    void updateWeights(const unsigned numEvents);
    
    /// Empty constructor
    RProp();

    /**
     * @brief Copy constructor
     *
     * This constructor should be used to create a new network which is an
     * exactly copy of another network.
     *
     * @param[in] net The network that we will copy the parameters from.
     **/
    RProp(const RProp &net);

    /**
     * @brief Constructor taking the parameters for a matlab net structure.
     *
     * This constructor should be called when the network parameters are
     * stored in a matlab network structure.
     * @param[in] netStr The Matlab network structure as returned by newff.
     * @param[in] trnParam The Matlab network train parameter
     *            (net.trainParam) structure.
     **/
    RProp( const NetConfHolder &net, 
           const MSG::Level msglevel, 
           const std::string &name);

    /**
     * @brief Returns a clone of the object.
     *
     * Returns a clone of the calling object. The clone is dynamically
     * allocated, so it must be released with delete at the end of its use.
     * @return A dynamically allocated clone of the calling object.
     **/
    virtual NeuralNetwork *clone(){ return new RProp(*this); }

    /**
     * @brief Copies all information that is used by training for the next
     *        epoch
     *
     * See overriden Backpropagation version for more information.
     **/
    virtual void copyNeededTrainingInfo(const Backpropagation &net) override;

    /**
     * @brief Fast copies all information that is used by training for the next
     *        epoch
     *
     * See overriden Backpropagation version for more information.
     **/
    virtual void copyNeededTrainingInfoFast(const Backpropagation &net) override;

    /**
     * @brief Class destructor.
     * Releases all the dynamically allocated memory used by the class, so
     * the user does not need to worry about dynamically allocated memory.
     **/
    virtual ~RProp();
    
    
    /**
     * @brief Gives the neural network information.
     * This method prints information about the neural
     * network. This method sould complement the information given by the
     * base class.
     * @see TuningTool::NeuralNetwork#showInfo 
     **/
    virtual void showInfo() const;

    /**
     * @brief Copy the status from the passing network.
     *
     * This method will make a deep copy of all attributes from the passing
     * network, making them exactly equal. This method <b>does not</b>
     * allocate any memory for the calling object. The space for weights and
     * bias info must have been previously created. 
     *
     * @param[in] net The network from where to copy the data from.
     **/
    RProp& operator=(const RProp &net);
};
}

#endif
